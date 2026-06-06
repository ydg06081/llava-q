from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LabConfig:
    model: dict[str, Any]
    data: dict[str, Any]
    train: dict[str, Any]
    output: dict[str, Any]


def load_config(path: str | Path) -> LabConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return LabConfig(
        model=raw.get("model", {}),
        data=raw.get("data", {}),
        train=raw.get("train", {}),
        output=raw.get("output", {}),
    )
