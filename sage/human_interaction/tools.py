from typing import Type
from dataclasses import dataclass, field
from datetime import date
from sage.retrieval.memory_bank import MemoryBank
from sage.base import SAGEBaseTool, BaseToolConfig, BaseConfig


@dataclass
class HumanInteractionToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: HumanInteractionTool)
    name: str = "human_interaction_tool"
    description: str = """Use this tool to communicate with the user. This
    can be to interact with the user to ask for more information on a topic or
    clarification on a previously requested command. Pass the query in a json with
    key "query".
    """


class HumanInteractionTool(SAGEBaseTool):
    """
    This tool is provided to the Coordinator to help it communicate with the
    user. This can be in numerous usecases like getting feedback, asking for more
    information or just having a conversation. The interaction can be done
    both via text or speech depending on the configuration setup.
    """

    memory: MemoryBank = None

    def setup(self, config: HumanInteractionToolConfig, memory=None):
        """
        Setup the AudioReader class
        """
        self.memory = memory

    @staticmethod
    def _update_stats(stats: dict[str, int] | None, key: str) -> None:
        if stats is None:
            return
        stats[key] = stats.get(key, 0) + 1

    def _prompt_user(self) -> tuple[str, str]:
        text_input = input("\nType (<username> : <command>) >> ")
        user_name, command = text_input.split(":")
        user_name = user_name.strip().lower()
        command = command.strip()

        if self.memory:
            self.memory.add_query(user_name, command, str(date.today()))

        return command, user_name

    def _run(self, dummy_string):
        """
        Returns the username and command utterance.
        """

        global_config = getattr(BaseConfig, "global_config", None)
        current_case = (
            getattr(global_config, "current_test_case", None) if global_config else None
        )
        current_types = (
            set(getattr(global_config, "current_test_types", []))
            if global_config
            else set()
        )
        stats = (
            getattr(global_config, "human_interaction_stats", None)
            if global_config
            else None
        )

        tracking_active = current_case is not None
        case_allows_human = tracking_active and ("human_interaction" in current_types)

        if tracking_active and not case_allows_human:
            self._update_stats(stats, "failure")
            return "", ""

        command, user_name = self._prompt_user()

        if tracking_active and case_allows_human:
            self._update_stats(stats, "success")

        return command, user_name


if __name__ == "__main__":
    interaction_config = HumanInteractionToolConfig()
    interaction_tool = interaction_config.instantiate()
