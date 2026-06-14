from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llava_mini.config import load_config
from llava_mini.data.collator import LlavaCollator
from llava_mini.data.dataset import JsonlLlavaDataset
from llava_mini.model.llava_qwen import LlavaQwenForCausalLM
from llava_mini.model.vision_tower import ClipVisionTower


def append_metric(path: str | Path, step: int, loss: float) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")


def run_synthetic_overfit(out_dir: str | Path, steps: int = 20) -> Path:
    metrics_path = Path(out_dir) / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    torch.manual_seed(42)
    layer = torch.nn.Linear(2, 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    x = torch.ones(4, 2)
    y = torch.ones(4, 1)
    for step in range(steps):
        loss = torch.nn.functional.mse_loss(layer(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        append_metric(metrics_path, step=step, loss=loss.item())
    return metrics_path


def run_real_overfit(
    config_path: str | Path,
    out_dir: str | Path,
    train_mode: str = "full",
) -> Path:
    if train_mode not in ("full", "projector"):
        raise ValueError(f"`train_mode` must be 'full' or 'projector'; got {train_mode!r}.")
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision = ClipVisionTower(cfg.model["vision_tower"], freeze=cfg.model.get("freeze_vision", True)).to(
        device
    )
    model = LlavaQwenForCausalLM(
        language_model_name=cfg.model["language_model"],
        vision_dim=vision.hidden_size,
    ).to(device)
    # The model owns its tokenizer and keeps `<image>` plus embeddings in sync.
    tokenizer = model.tokenizer
    if train_mode == "projector":
        for param in model.language_model.parameters():
            param.requires_grad_(False)

    collator = LlavaCollator(tokenizer=tokenizer, image_processor=vision.image_processor)
    dataset = JsonlLlavaDataset(cfg.data["train_jsonl"])
    loader = DataLoader(dataset, batch_size=cfg.train.get("batch_size", 1), shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    opt = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.train.get("learning_rate", 2e-5),
    )
    metrics_path = Path(out_dir) / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    model.train()
    for step in range(cfg.train.get("max_steps", 20)):
        pixel_values = batch["pixel_values"].to(device)
        with torch.no_grad():
            image_features = vision.encode_images(pixel_values)
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            image_features=image_features,
            image_positions=batch["image_positions"],
        )
        loss = outputs.loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        append_metric(metrics_path, step=step, loss=loss.item())
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen15b_clip.yaml")
    parser.add_argument("--out", default="outputs/overfit")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--real", action="store_true", help="Load CLIP/Qwen and run a real one-batch overfit.")
    args = parser.parse_args()

    if args.real:
        metrics_path = run_real_overfit(args.config, args.out)
    else:
        metrics_path = run_synthetic_overfit(args.out, steps=args.steps)
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
