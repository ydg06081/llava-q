from __future__ import annotations

import argparse

from llava_mini.train.overfit_one_batch import run_real_overfit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Projector-only training (LLM and vision frozen) one-batch overfit."
    )
    parser.add_argument("--config", default="configs/qwen15b_clip.yaml")
    parser.add_argument("--out", default="outputs/train-projector")
    args = parser.parse_args()
    metrics_path = run_real_overfit(args.config, args.out, train_mode="projector")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
