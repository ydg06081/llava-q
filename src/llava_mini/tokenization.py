from __future__ import annotations

from typing import Any

from llava_mini.constants import IMAGE_TOKEN


def ensure_image_token(tokenizer: Any) -> int:
    """Register ``<image>`` as a single special token (idempotent).

    Returns the number of tokens actually added (0 if it was already present).
    Callers that own model embeddings must call ``resize_token_embeddings`` when
    this returns a non-zero value so the new id has a matching embedding row.
    """
    added = tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if token_id is None or token_id < 0:
        raise ValueError(f"Tokenizer failed to register {IMAGE_TOKEN}.")
    return int(added)
