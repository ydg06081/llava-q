#!/usr/bin/env bash
set -euo pipefail

uv run python -m llava_mini.train.overfit_one_batch --out outputs/overfit --steps 20
