from __future__ import annotations

import argparse

from llava_mini.train.overfit_one_batch import run_real_overfit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full fine-tune (projector + LLM, vision frozen) one-batch overfit."
    )
    parser.add_argument("--config", default="configs/qwen15b_clip.yaml")
    parser.add_argument("--out", default="outputs/train-full")
    args = parser.parse_args()
    metrics_path = run_real_overfit(args.config, args.out, train_mode="full")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
