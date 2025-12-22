"""
A smartthings tool to disambiguate which device a user is referring using a generic text command.

The tool uses a VLM (CLIP) to find the image that has the closest embedding to the text command. The tool requires an image folder path that contains images of the target devices, and the clip embbeding model used. The tool will eliminate devices for which is does not have an image, or that have been filtered out by the smartthings LLM agent (list of candidate devices)
"""
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Type

import nltk
import numpy as np
import open_clip as clip
import torch
from PIL import Image

from sage.base import BaseToolConfig
from sage.base import SAGEBaseTool
from sage.utils.common import parse_json


def is_noun(pos):
    return pos[:2] == "NN"


def extract_nouns(text: str) -> str:
    """Filter a string to keep only the nouns"""
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")

    tokenized = nltk.word_tokenize(text)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

    return " ".join(word for word in nouns)


def get_image_embeds(
    image_input: torch.Tensor, model: clip.CLIP, device: str = "cpu"
) -> torch.Tensor:
    """Get image embeddings"""
    image_embeds = []
    batch_size = 100

    for i in range(0, len(image_input), batch_size):
        print(i, image_input.shape[0])
        with torch.no_grad():
            cuda_input = image_input[i : i + batch_size].to(device)
            image_embeds.append(model.encode_image(cuda_input).cpu().float())
            del cuda_input
    image_embeds = torch.cat(image_embeds)
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    image_embeds = image_embeds.cpu().numpy()

    return image_embeds


def get_text_embeds(
    text: list[str], model: clip.CLIP, device: str = "cpu"
) -> torch.Tensor:
    """Get text embeddings"""
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens).cpu().float()
    del text_tokens
    text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().numpy()

    return text_embeds


def clean_embeds(
    embeds: np.ndarray,
    drop_above: float = 0.3,
    drop_below: float = 0,
    renorm_after_drop: bool = True,
) -> torch.Tensor:
    """
    Clean up embeddings, usually by removing large dimensions.

    For an explanation of why you might want to do this, see https://arxiv.org/abs/2302.07931 section titled
    "dropping large embedding dimensions".
    """
    embeds = embeds.copy()
    embeds[np.abs(embeds) > drop_above] = 0
    embeds[np.abs(embeds) < drop_below] = 0

    if renorm_after_drop:
        embeds /= np.linalg.norm(embeds, axis=-1, keepdims=True)

    return embeds


class VlmDeviceDetector:
    """Use a VLM to find the closest match between a user query (text) and available devices (images)."""

    def __init__(self, image_folder: str):
        # 尝试检测 CUDA 是否真正可用，如果不可用则回退到 CPU
        self.device = self._get_safe_device()
        self.image_folder = image_folder
        self.model, _, self.preprocess = clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        # 先加载到 CPU，如果需要再移动到 GPU（避免初始化时的 CUDA 错误）
        self.model = self.model.to("cpu")
        # 如果设备是 CUDA，尝试移动到 GPU，如果失败则回退到 CPU
        if self.device == "cuda":
            try:
                self.model = self.model.to(self.device)
                # 测试 CUDA 是否真正可用（执行一个简单的操作）
                test_tensor = torch.zeros(1).to(self.device)
                del test_tensor
            except Exception as exc:
                # CUDA 不可用，回退到 CPU
                import warnings
                warnings.warn(f"CUDA initialization failed, falling back to CPU: {exc}")
                self.device = "cpu"
                self.model = self.model.to("cpu")
    
    def _get_safe_device(self) -> str:
        """安全地检测可用的设备，优先使用 CUDA，如果不可用则使用 CPU"""
        if not torch.cuda.is_available():
            return "cpu"
        # 尝试检测 CUDA 是否真正可用（不仅仅是 is_available()）
        try:
            # 创建一个测试张量来验证 CUDA 是否真正可用
            test = torch.zeros(1).cuda()
            del test
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            # CUDA 不可用，回退到 CPU
            return "cpu"

    def identify_device(self, command: str) -> str:
        """
        Identify devices based on command string

        The format of the command string should be the same as that described in the DeviceDisambiguationToolConfig.
        """

        winner, _ = self.identify_device_with_scores(command)
        return winner

    def identify_device_with_scores(self, command: str) -> tuple[str, list[tuple[str, float]]]:
        """
        Identify devices and also return similarity scores.

        Returns:
            (winner_device_id, [(device_id, score), ...] sorted by score desc)
            If no valid devices, returns (error_message, []).
        """

        image_dict = self.get_images()

        attr_spec = parse_json(command)
        attr_spec["disambiguation_information"] = extract_nouns(
            attr_spec["disambiguation_information"]
        )

        if attr_spec is None:
            return "Invalid JSON format. The input should be a json string with 2 keys: devices (list of guid strings) and disambiguation_information (str): objects or descriptions of device surroundings."

        # Some LLMs (e.g., LEMUR) query the device disambiguation tool with only
        # one device ID in the list. Instead of returning an error, just return
        # the ID of the device.

        if len(attr_spec["devices"]) == 1:
            return attr_spec["devices"][0], [(attr_spec["devices"][0], 1.0)]

        device_list = sorted(
            self.select_devices(attr_spec["devices"], list(image_dict.keys()))
        )

        if not device_list:
            return (
                "None of the devices you listed were found. Avoid coming up with fake device IDs and consider checking the API planner first. Correct device ID is a guid string not a generic name (e.g. an incorrect name is device1)?",
                [],
            )

        with torch.no_grad():
            try:
                text_embeds = get_text_embeds(
                    attr_spec["disambiguation_information"], self.model, self.device
                )
                images = [self.preprocess(image_dict[d]["image"]) for d in device_list]
                images = torch.stack(images)
                image_embeds = clean_embeds(
                    get_image_embeds(images, self.model, self.device)
                )

                sims = (text_embeds @ image_embeds.T).squeeze()
                ranked_indices = np.argsort(sims)[::-1]
                ranked = [(device_list[i], float(sims[i])) for i in ranked_indices]
                winner = ranked[0][0] if ranked else ""
                return winner, ranked
            except RuntimeError as exc:
                # 捕获 CUDA 运行时错误，尝试回退到 CPU
                if "cuda" in str(exc).lower() or "CUDA" in str(exc):
                    import warnings
                    warnings.warn(f"CUDA error during inference, falling back to CPU: {exc}")
                    # 将模型移动到 CPU 并重试
                    original_device = self.device
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    try:
                        text_embeds = get_text_embeds(
                            attr_spec["disambiguation_information"], self.model, self.device
                        )
                        images = [self.preprocess(image_dict[d]["image"]) for d in device_list]
                        images = torch.stack(images)
                        image_embeds = clean_embeds(
                            get_image_embeds(images, self.model, self.device)
                        )

                        sims = (text_embeds @ image_embeds.T).squeeze()
                        ranked_indices = np.argsort(sims)[::-1]
                        ranked = [(device_list[i], float(sims[i])) for i in ranked_indices]
                        winner = ranked[0][0] if ranked else ""
                        return winner, ranked
                    except Exception as cpu_exc:
                        return f"VLM inference failed on both CUDA and CPU: {cpu_exc}", []
                else:
                    # 非 CUDA 错误，直接抛出
                    raise

    def get_images(self) -> dict[str, Any]:
        """Load images from image folder."""
        image_dict = {}
        file_list = [
            os.path.join(self.image_folder, file)
            for file in os.listdir(self.image_folder)
        ]

        for filepath in file_list:
            # 使用 os.path.basename 和 os.path.splitext 来跨平台兼容地提取文件名（不含扩展名）
            filename = os.path.basename(filepath)
            name = os.path.splitext(filename)[0]
            image_dict[name] = {}
            image = Image.open(filepath)
            image_dict[name]["filepath"] = filepath
            image_dict[name]["image"] = image

        return image_dict

    def select_devices(
        self, llm_devices: list[str], real_devices: list[str]
    ) -> list[str]:
        """
        Applies a fuzzy matching based on the number of matching characters.

        Necessary because LLMs sometimes make mistakes copying the IDs.
        """
        threshold = 0.7
        out = []

        for llm_str in llm_devices:
            for real_str in real_devices:
                count = 0

                for c_llm, c_real in zip(llm_str, real_str):
                    if c_llm == c_real:
                        count += 1
                frac = count / len(real_str)
                # print(frac)

                if frac > threshold:
                    out.append(real_str)

        return out


@dataclass
class DeviceDisambiguationToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: DeviceDisambiguationTool)
    name: str = "device_disambiguation"
    description: str = """
Use this to select the right device from a list of candidate devices based on the location and/or the surroundings description. Do not use this tool without specific description of the surroundings.
Use this tool for one device type at a time.
Input to the tool should be a json string with 2 keys:
devices (list of guid strings).
disambiguation_information (str): objects or descriptions of device surroundings. Do not include the name of the device type.
"""
    # Update this path to the folder containing images of the devices if using own images
    image_folder: str = f"{os.getenv('SMARTHOME_ROOT')}/user_device_images"


class DeviceDisambiguationTool(SAGEBaseTool):
    """Wraps the VLM-based device disambiguation pipeline as a Tool"""

    detector: VlmDeviceDetector = None

    def setup(self, config: DeviceDisambiguationToolConfig) -> None:
        if config.global_config.test_id is not None:
            self.detector = VlmDeviceDetector(f"{os.getenv('SMARTHOME_ROOT')}/sage/testing/assets/images")
        else:
            self.detector = VlmDeviceDetector(config.image_folder)

    def _run(self, query: str) -> str:
        return self.detector.identify_device(query)
