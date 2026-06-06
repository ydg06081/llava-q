import torch

from llava_mini.model.llava_qwen import expand_for_image_tokens, splice_image_embeddings


def test_splice_image_embeddings_replaces_placeholder():
    text_embeds = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3)
    image_embeds = torch.ones(1, 2, 3)
    output = splice_image_embeddings(text_embeds, image_embeds, image_positions=[1])

    assert output.shape == (1, 5, 3)
    assert torch.all(output[:, 1:3] == 1)
    assert torch.equal(output[:, 0], text_embeds[:, 0])
    assert torch.equal(output[:, 3:], text_embeds[:, 2:])


def test_expand_for_image_tokens_replaces_placeholder_label():
    labels = torch.tensor([[10, -100, 20, 21]])
    expanded = expand_for_image_tokens(labels, image_token_count=2, image_positions=[1], fill_value=-100)

    assert expanded.tolist() == [[10, -100, -100, 20, 21]]
