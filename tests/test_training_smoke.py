import json

import torch

from llava_mini.train.overfit_one_batch import append_metric, run_synthetic_overfit


def test_append_metric_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_metric(path, step=1, loss=0.5)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {"step": 1, "loss": 0.5}


def test_optimizer_can_reduce_simple_loss():
    layer = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(layer.parameters(), lr=0.1)
    x = torch.ones(4, 2)
    y = torch.ones(4, 1)
    first = None
    last = None
    for step in range(20):
        loss = torch.nn.functional.mse_loss(layer(x), y)
        if step == 0:
            first = loss.item()
        last = loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert last < first


def test_synthetic_overfit_metrics_decrease(tmp_path):
    metrics = run_synthetic_overfit(tmp_path / "overfit", steps=8)
    losses = [
        json.loads(line)["loss"]
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert losses[-1] < losses[0]
