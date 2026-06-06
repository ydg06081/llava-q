from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from llava_mini.constants import IGNORE_INDEX, IMAGE_TOKEN
from llava_mini.data.schema import LlavaRecord, render_prompt


def build_labels(input_ids: list[int], answer_start: int) -> list[int]:
    if answer_start < 0 or answer_start > len(input_ids):
        raise ValueError("`answer_start` must be inside the input sequence.")
    return [IGNORE_INDEX] * answer_start + list(input_ids[answer_start:])


def find_image_token_position(input_ids: list[int], image_token_id: int) -> int:
    positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
    if len(positions) != 1:
        raise ValueError(
            f"Expected exactly one {IMAGE_TOKEN} token id {image_token_id}; found {len(positions)}."
        )
    return positions[0]


@dataclass(frozen=True)
class CollatedSampleTrace:
    prompt_text: str
    answer_text: str
    prompt_token_count: int
    answer_token_count: int
    image_token_position: int
    labels: list[int]


class LlavaCollator:
    def __init__(self, tokenizer: Any, image_processor: Any | None = None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def tokenize_record(self, record: LlavaRecord) -> tuple[list[int], CollatedSampleTrace]:
        prompt = render_prompt(record.question)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(record.answer, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + answer_ids
        labels = build_labels(input_ids, len(prompt_ids))
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        trace = CollatedSampleTrace(
            prompt_text=prompt,
            answer_text=record.answer,
            prompt_token_count=len(prompt_ids),
            answer_token_count=len(answer_ids),
            image_token_position=find_image_token_position(input_ids, image_token_id),
            labels=labels,
        )
        return input_ids, trace

    def __call__(self, records: list[LlavaRecord]) -> dict[str, Any]:
        tokenized = [self.tokenize_record(record) for record in records]
        input_ids_list = [ids for ids, _trace in tokenized]
        traces = [trace for _ids, trace in tokenized]
        labels_list = [trace.labels for trace in traces]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = []
        labels = []
        attention_mask = []
        for ids, label_ids in zip(input_ids_list, labels_list, strict=True):
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(label_ids + [IGNORE_INDEX] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "image_positions": [trace.image_token_position for trace in traces],
            "traces": traces,
        }

        if self.image_processor is not None:
            images = [self.load_image(record.image) for record in records]
            batch.update(self.image_processor(images=images, return_tensors="pt"))
        return batch

    def load_image(self, path: str | Path) -> Image.Image:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")
