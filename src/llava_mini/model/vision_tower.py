from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModel


class ClipVisionTower(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        if freeze:
            self.freeze()

    @property
    def hidden_size(self) -> int:
        return int(self.vision_model.config.hidden_size)

    def freeze(self) -> None:
        self.vision_model.eval()
        for param in self.vision_model.parameters():
            param.requires_grad_(False)

    def preprocess(self, images: Any) -> dict[str, torch.Tensor]:
        return self.image_processor(images=images, return_tensors="pt")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        # LLaVA selects the penultimate (second-to-last) hidden layer, then drops the
        # CLS token so each remaining token corresponds to an image patch.
        return outputs.hidden_states[-2][:, 1:, :]
