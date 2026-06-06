from llava_mini.visualize.sample_trace import build_demo_trace, render_trace_html


def test_render_trace_html_contains_key_sections():
    html = render_trace_html(
        {
            "image": "toy.png",
            "prompt_tokens": ["<image>", "What"],
            "answer_tokens": ["square"],
            "labels": [-100, -100, 123],
            "shapes": {"projected": [1, 4, 896]},
        }
    )
    assert "label masking" in html.lower()
    assert "projected" in html


def test_build_demo_trace_contains_image_token():
    trace = build_demo_trace()
    assert "<image>" in trace["prompt_tokens"]
