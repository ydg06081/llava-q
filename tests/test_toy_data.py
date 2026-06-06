from pathlib import Path

from scripts.prepare_toy_data import create_toy_dataset


def test_create_toy_dataset(tmp_path: Path):
    out = tmp_path / "toy"
    jsonl_path = create_toy_dataset(out)

    assert jsonl_path.exists()
    rows = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) >= 3
    assert (out / "images").exists()
