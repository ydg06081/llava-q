from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


TOY_RECORDS = [
    {
        "file": "red_square.png",
        "question": "What color is the square?",
        "answer": "The square is red.",
        "color": (220, 40, 40),
        "shape": "square",
    },
    {
        "file": "blue_circle.png",
        "question": "What shape is shown?",
        "answer": "A blue circle is shown.",
        "color": (40, 90, 220),
        "shape": "circle",
    },
    {
        "file": "green_triangle.png",
        "question": "What is in the image?",
        "answer": "There is a green triangle.",
        "color": (40, 160, 90),
        "shape": "triangle",
    },
]


def _draw_shape(path: Path, color: tuple[int, int, int], shape: str) -> None:
    image = Image.new("RGB", (224, 224), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    if shape == "square":
        draw.rectangle((54, 54, 170, 170), fill=color)
    elif shape == "circle":
        draw.ellipse((48, 48, 176, 176), fill=color)
    elif shape == "triangle":
        draw.polygon([(112, 42), (42, 180), (182, 180)], fill=color)
    else:
        raise ValueError(f"Unsupported toy shape: {shape}")
    image.save(path)


def create_toy_dataset(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in TOY_RECORDS:
        image_path = image_dir / record["file"]
        _draw_shape(image_path, record["color"], record["shape"])
        rows.append(
            {
                "image": str(image_path),
                "question": record["question"],
                "answer": record["answer"],
            }
        )

    jsonl_path = out / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy", help="Output directory for toy data.")
    args = parser.parse_args()
    path = create_toy_dataset(args.out)
    print(f"Wrote toy dataset: {path}")


if __name__ == "__main__":
    main()


# 실전에서는 TOY_RECORDS처럼 코드 안에 QA를 직접 박아두지 않고,
# 이미지 폴더와 annotation/caption/QA 파일을 읽어서 아래처럼 JSONL로 변환합니다.
#
# def convert_my_dataset(image_dir: str | Path, annotation_path: str | Path, out_jsonl: str | Path) -> Path:
#     image_dir = Path(image_dir)
#     annotation_path = Path(annotation_path)
#     out_jsonl = Path(out_jsonl)
#     out_jsonl.parent.mkdir(parents=True, exist_ok=True)
#
#     rows = []
#     # 예: annotation_path가 CSV/JSON/DB export라고 가정합니다.
#     # 각 annotation에는 image filename, question, answer가 들어 있어야 합니다.
#     annotations = load_my_annotations(annotation_path)
#     for item in annotations:
#         image_path = image_dir / item["image_file"]
#         if not image_path.exists():
#             raise FileNotFoundError(f"Image not found: {image_path}")
#         rows.append(
#             {
#                 "image": str(image_path),
#                 "question": item["question"],
#                 "answer": item["answer"],
#             }
#         )
#
#     with out_jsonl.open("w", encoding="utf-8") as f:
#         for row in rows:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")
#     return out_jsonl
