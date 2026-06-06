import torch

from llava_mini.model.projector import VisionProjector


def test_projector_maps_vision_dim_to_text_dim():
    projector = VisionProjector(vision_dim=8, text_dim=16)
    x = torch.randn(2, 5, 8)
    y = projector(x)
    assert y.shape == (2, 5, 16)
