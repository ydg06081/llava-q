from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava_mini.model.projector import VisionProjector


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

    rows = []
    seq_len = text_embeds.shape[1]
    for batch_idx, position in enumerate(image_positions):
        if position < 0 or position >= seq_len:
            raise ValueError(f"Image position {position} is outside sequence length {seq_len}.")
        before = text_embeds[batch_idx, :position]
        after = text_embeds[batch_idx, position + 1 :]
        rows.append(torch.cat([before, image_embeds[batch_idx], after], dim=0))

    lengths = {row.shape[0] for row in rows}
    if len(lengths) != 1:
        raise ValueError("All spliced rows must have the same length before batching.")
    return torch.stack(rows, dim=0)


@dataclass(frozen=True)
class MultimodalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor
    image_positions: list[int]


class LlavaQwenForCausalLM(nn.Module):
    def __init__(
        self,
        language_model_name: str,
        vision_dim: int,
        projector: VisionProjector | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        text_dim = int(self.language_model.config.hidden_size)
        self.projector = projector or VisionProjector(vision_dim=vision_dim, text_dim=text_dim)

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


def expand_for_image_tokens(
    values: torch.Tensor,
    image_token_count: int,
    image_positions: list[int],
    fill_value: int,
) -> torch.Tensor:
    rows = []
    for batch_idx, position in enumerate(image_positions):
        fill = values.new_full((image_token_count,), fill_value)
        rows.append(torch.cat([values[batch_idx, :position], fill, values[batch_idx, position + 1 :]]))
    return torch.stack(rows, dim=0)
