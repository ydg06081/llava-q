from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from llava_mini.constants import IGNORE_INDEX
from llava_mini.data.schema import LlavaRecord, render_prompt


def build_labels(input_ids: list[int], answer_start: int) -> list[int]:
    if answer_start < 0 or answer_start > len(input_ids):
        raise ValueError("`answer_start` must be inside the input sequence.")
    return [IGNORE_INDEX] * answer_start + list(input_ids[answer_start:])


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
        trace = CollatedSampleTrace(
            prompt_text=prompt,
            answer_text=record.answer,
            prompt_token_count=len(prompt_ids),
            answer_token_count=len(answer_ids),
            image_token_position=prompt.index("<image>"),
            labels=labels,
        )
        return input_ids, trace

    def load_image(self, path: str | Path) -> Image.Image:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")
