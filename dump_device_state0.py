"""
Utility: dump device_state0.pkl to CSV (sage/testing/device_state0_dump.csv)
No external deps on project modules; reads pickle directly.
"""
import csv
import json
import pickle as pkl
from pathlib import Path


def load_device_names() -> dict:
    doc_path = Path("external_api_docs/cached_test_docmanager.json")
    data = json.loads(doc_path.read_text(encoding="utf-8"))
    return data.get("device_names", {}) if isinstance(data, dict) else {}


def load_device_state() -> dict:
    pkl_path = Path("sage/testing/device_state0.pkl")
    data = pkl.loads(pkl_path.read_bytes())
    # pkl is list of [{"device_id":..., "components": {...}}, ...]
    return {item[0]["device_id"]: item[0]["components"] for item in data}


def main() -> None:
    device_names = load_device_names()
    state = load_device_state()

    rows = []
    for did, comps in state.items():
        name = device_names.get(did, did)
        for comp_name, comp in comps.items():
            for cap_name, cap in comp.items():
                for attr_name, attr in cap.items():
                    val = attr["value"] if isinstance(attr, dict) and "value" in attr else attr
                    rows.append(
                        {
                            "device_id": did,
                            "device_name": name,
                            "component": comp_name,
                            "capability": cap_name,
                            "attribute": attr_name,
                            "value": val,
                        }
                    )

    out = Path("sage/testing/device_state0_dump.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["device_id", "device_name", "component", "capability", "attribute", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

