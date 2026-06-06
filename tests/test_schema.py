import pytest

from llava_mini.data.schema import LlavaRecord, render_prompt


def test_record_requires_image_question_answer():
    record = LlavaRecord(image="x.png", question="What?", answer="A square.")
    assert record.image == "x.png"


def test_render_prompt_contains_one_image_placeholder():
    text = render_prompt("What is shown?")
    assert text.count("<image>") == 1


def test_render_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        render_prompt("")
