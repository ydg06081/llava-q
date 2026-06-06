from __future__ import annotations

from dataclasses import dataclass

from llava_mini.constants import IMAGE_TOKEN


@dataclass(frozen=True)
class LlavaRecord:
    image: str
    question: str
    answer: str

    def __post_init__(self) -> None:
        for field_name in ("image", "question", "answer"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{field_name}` must be a non-empty string.")


def render_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("`question` must be a non-empty string.")
    prompt = f"<|im_start|>user\n{IMAGE_TOKEN}\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"
    if prompt.count(IMAGE_TOKEN) != 1:
        raise ValueError(f"Rendered prompt must contain exactly one {IMAGE_TOKEN} placeholder.")
    return prompt
