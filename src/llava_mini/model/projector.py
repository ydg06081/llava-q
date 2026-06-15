from __future__ import annotations

import torch
from torch import nn


class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        if vision_dim <= 0 or text_dim <= 0:
            raise ValueError("`vision_dim` and `text_dim` must be positive.")
        self.proj = nn.Linear(vision_dim, text_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 3:
            raise ValueError(
                "`image_features` must have shape (batch, patches, vision_dim); "
                f"got {tuple(image_features.shape)}."
            )
        return self.proj(image_features)
#단순하게 vision_dim에서 text_dim으로 선형 변환하는 레이어.
#image_features는 (batch, patches, vision_dim) 형태의 텐서여야 함.
