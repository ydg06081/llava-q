from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from llava_mini.data.schema import LlavaRecord


class JsonlLlavaDataset(Dataset[LlavaRecord]):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset JSONL not found: {self.path}")
        self.records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                try:
                    self.records.append(
                        LlavaRecord(
                            image=raw["image"],
                            question=raw["question"],
                            answer=raw["answer"],
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"Missing field {exc} on line {line_no} of {self.path}") from exc
        if not self.records:
            raise ValueError(f"Dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LlavaRecord:
        return self.records[index]
