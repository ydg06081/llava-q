from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava_mini.model.projector import VisionProjector
from llava_mini.tokenization import ensure_image_token


def _splice_rows(
    values: torch.Tensor,
    image_positions: list[int],
    make_block: Callable[[int], torch.Tensor],
) -> torch.Tensor:
    """Replace one placeholder slot per row with a row-specific block.

    For each batch row the token at ``image_positions[row]`` is dropped and the
    tensor returned by ``make_block(row)`` is inserted in its place. Used both to
    splice image embeddings into the text embeddings and to expand the 1-D
    label / attention-mask rows to match.
    """
    rows = []
    for batch_idx, position in enumerate(image_positions):
        before = values[batch_idx, :position]
        after = values[batch_idx, position + 1 :]
        rows.append(torch.cat([before, make_block(batch_idx), after], dim=0))
    return torch.stack(rows, dim=0)


def splice_image_embeddings(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    image_positions: list[int],
) -> torch.Tensor:
    if text_embeds.ndim != 3 or image_embeds.ndim != 3:
        raise ValueError("`text_embeds` and `image_embeds` must be 3D tensors.")
    if text_embeds.shape[0] != image_embeds.shape[0]:
        raise ValueError("Text and image batch sizes must match.")
    if text_embeds.shape[2] != image_embeds.shape[2]:
        raise ValueError("Text and image embedding dimensions must match.")
    if len(image_positions) != text_embeds.shape[0]:
        raise ValueError("One image placeholder position is required for each batch row.")

    seq_len = text_embeds.shape[1]
    for position in image_positions:
        if position < 0 or position >= seq_len:
            raise ValueError(f"Image position {position} is outside sequence length {seq_len}.")
    return _splice_rows(text_embeds, image_positions, lambda batch_idx: image_embeds[batch_idx])


def expand_for_image_tokens(
    values: torch.Tensor,
    image_token_count: int,
    image_positions: list[int],
    fill_value: int,
) -> torch.Tensor:
    return _splice_rows(
        values,
        image_positions,
        lambda _batch_idx: values.new_full((image_token_count,), fill_value),
    )


class LlavaQwenForCausalLM(nn.Module):
    def __init__(
        self,
        language_model_name: str,
        vision_dim: int,
        projector: VisionProjector | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        # Load in float32 so the LM matches the float32 CLIP/projector path; Qwen
        # ships bfloat16 weights, which would otherwise mismatch at matmul time.
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, dtype=torch.float32
        )
        self.prepare_tokenizer()
        text_dim = int(self.language_model.config.hidden_size)
        self.projector = projector or VisionProjector(vision_dim=vision_dim, text_dim=text_dim)

    def prepare_tokenizer(self) -> None:
        """Register `<image>` and keep the tokenizer and embeddings in sync."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        added = ensure_image_token(self.tokenizer)
        if added:
            self.language_model.resize_token_embeddings(len(self.tokenizer))

    @property
    def hidden_size(self) -> int:
        return int(self.language_model.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        image_positions: list[int],
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_embeds = self.projector(image_features)
        inputs_embeds = splice_image_embeddings(text_embeds, image_embeds, image_positions)

        model_kwargs = {"inputs_embeds": inputs_embeds}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = expand_for_image_tokens(
                attention_mask, image_embeds.shape[1], image_positions, fill_value=1
            )
        if labels is not None:
            model_kwargs["labels"] = expand_for_image_tokens(
                labels, image_embeds.shape[1], image_positions, fill_value=-100
            )
        return self.language_model(**model_kwargs)
