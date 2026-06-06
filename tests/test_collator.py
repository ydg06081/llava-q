from llava_mini.constants import IGNORE_INDEX
from llava_mini.data.collator import find_image_token_position, build_labels


def test_build_labels_masks_prompt_tokens():
    input_ids = [10, 11, 12, 13, 14]
    answer_start = 3
    labels = build_labels(input_ids, answer_start)

    assert labels[:answer_start] == [IGNORE_INDEX] * answer_start
    assert labels[answer_start:] == input_ids[answer_start:]


def test_find_image_token_position_requires_exactly_one_token():
    assert find_image_token_position([1, 2, 99, 3], image_token_id=99) == 2
