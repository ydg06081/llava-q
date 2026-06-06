# LLaVA Mini Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small LLaVA 1 style learning lab using CLIP plus `Qwen/Qwen2.5-0.5B-Instruct`, with visible data transforms, image-token insertion, label masking, and real training loops.

**Architecture:** Use Hugging Face Transformers for CLIP and Qwen, but implement the LLaVA bridge, prompt rendering, multimodal embedding splice, data collator, traces, and training loops directly. Keep CLIP frozen by default, train a projector, and allow Qwen full fine-tuning for the first educational path.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Accelerate, Datasets-style JSONL loading, pytest, matplotlib/HTML trace output.

---

## File Structure

- Create `pyproject.toml`: project metadata, dependencies, pytest config, ruff config.
- Create `README.md`: Korean quickstart, learning goals, training commands.
- Create `configs/qwen05b_clip.yaml`: default model, data, train, and visualization settings.
- Create `src/llava_mini/__init__.py`: package marker.
- Create `src/llava_mini/config.py`: typed config loading from YAML.
- Create `src/llava_mini/constants.py`: `<image>` token and label mask constants.
- Create `src/llava_mini/data/schema.py`: JSONL record validation and prompt rendering.
- Create `src/llava_mini/data/dataset.py`: dataset class that loads image paths and records.
- Create `src/llava_mini/data/collator.py`: tokenizer, image processor, padding, label masking metadata.
- Create `src/llava_mini/model/vision_tower.py`: CLIP processor/model wrapper.
- Create `src/llava_mini/model/projector.py`: linear projector module.
- Create `src/llava_mini/model/llava_qwen.py`: multimodal forward and generate logic.
- Create `src/llava_mini/train/overfit_one_batch.py`: tiny training proof.
- Create `src/llava_mini/train/train_projector.py`: projector-only training entrypoint.
- Create `src/llava_mini/train/train_full.py`: projector plus Qwen full fine-tune entrypoint.
- Create `src/llava_mini/visualize/sample_trace.py`: sample trace exporter.
- Create `scripts/prepare_toy_data.py`: deterministic tiny image/question/answer JSONL generator.
- Create `scripts/run_overfit.sh`: convenience command for the first sanity run.
- Create tests under `tests/` for schema, collator masking, projector shape, multimodal splice, and toy data generation.

## Task 1: Project Skeleton And Toy Data

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/qwen05b_clip.yaml`
- Create: `src/llava_mini/__init__.py`
- Create: `src/llava_mini/constants.py`
- Create: `src/llava_mini/config.py`
- Create: `scripts/prepare_toy_data.py`
- Test: `tests/test_toy_data.py`

- [ ] **Step 1: Write failing toy data test**

```python
from pathlib import Path

from scripts.prepare_toy_data import create_toy_dataset


def test_create_toy_dataset(tmp_path: Path):
    out = tmp_path / "toy"
    jsonl_path = create_toy_dataset(out)

    assert jsonl_path.exists()
    rows = jsonl_path.read_text().strip().splitlines()
    assert len(rows) >= 3
    assert (out / "images").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_toy_data.py -v`

Expected: FAIL because `scripts.prepare_toy_data` does not exist.

- [ ] **Step 3: Implement skeleton and toy data generator**

Create minimal package files, config file, and a toy data script that writes a few simple generated PNG images and JSONL records with `image`, `question`, and `answer`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_toy_data.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md configs src scripts tests/test_toy_data.py
git commit -m "feat: scaffold llava mini lab"
```

## Task 2: Data Schema And Prompt Rendering

**Files:**
- Create: `src/llava_mini/data/schema.py`
- Create: `src/llava_mini/data/__init__.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_schema.py -v`

Expected: FAIL because schema module does not exist.

- [ ] **Step 3: Implement schema and prompt rendering**

Use a small dataclass. Validate non-empty fields. Render a Qwen chat-style prompt with exactly one `<image>` placeholder and no answer text in the prompt portion.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_schema.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llava_mini/data tests/test_schema.py
git commit -m "feat: add llava data schema"
```

## Task 3: Label Masking And Collator Metadata

**Files:**
- Create: `src/llava_mini/data/collator.py`
- Modify: `src/llava_mini/constants.py`
- Test: `tests/test_collator.py`

- [ ] **Step 1: Write failing masking tests**

```python
from llava_mini.constants import IGNORE_INDEX
from llava_mini.data.collator import build_labels


def test_build_labels_masks_prompt_tokens():
    input_ids = [10, 11, 12, 13, 14]
    answer_start = 3
    labels = build_labels(input_ids, answer_start)

    assert labels[:answer_start] == [IGNORE_INDEX] * answer_start
    assert labels[answer_start:] == input_ids[answer_start:]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_collator.py -v`

Expected: FAIL because `build_labels` does not exist.

- [ ] **Step 3: Implement masking helper and batch trace metadata**

Implement a pure helper first. Then add a collator class that records enough metadata to explain prompt token count, answer token count, image placeholder position, and label mask.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_collator.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llava_mini/constants.py src/llava_mini/data/collator.py tests/test_collator.py
git commit -m "feat: add label masking collator"
```

## Task 4: Vision Tower And Projector

**Files:**
- Create: `src/llava_mini/model/__init__.py`
- Create: `src/llava_mini/model/vision_tower.py`
- Create: `src/llava_mini/model/projector.py`
- Test: `tests/test_projector.py`

- [ ] **Step 1: Write failing projector shape test**

```python
import torch

from llava_mini.model.projector import VisionProjector


def test_projector_maps_vision_dim_to_text_dim():
    projector = VisionProjector(vision_dim=8, text_dim=16)
    x = torch.randn(2, 5, 8)
    y = projector(x)
    assert y.shape == (2, 5, 16)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_projector.py -v`

Expected: FAIL because model package does not exist.

- [ ] **Step 3: Implement projector and CLIP wrapper**

Projector should be simple `nn.Linear` by default. CLIP wrapper should expose `encode_images(pixel_values) -> patch_features` and freeze parameters when configured.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_projector.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llava_mini/model tests/test_projector.py
git commit -m "feat: add vision projector"
```

## Task 5: Multimodal Qwen Forward

**Files:**
- Create: `src/llava_mini/model/llava_qwen.py`
- Test: `tests/test_multimodal_splice.py`

- [ ] **Step 1: Write failing splice unit test**

Use a tiny fake embedding matrix and fake image embeddings to test pure splice logic without loading Qwen.

```python
import torch

from llava_mini.model.llava_qwen import splice_image_embeddings


def test_splice_image_embeddings_replaces_placeholder():
    text_embeds = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3)
    image_embeds = torch.ones(1, 2, 3)
    output = splice_image_embeddings(text_embeds, image_embeds, image_positions=[1])

    assert output.shape == (1, 5, 3)
    assert torch.all(output[:, 1:3] == 1)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_multimodal_splice.py -v`

Expected: FAIL because `splice_image_embeddings` does not exist.

- [ ] **Step 3: Implement splice helper and model wrapper**

Implement pure splice logic first. Then add `LlavaQwenForCausalLM` that loads Qwen, gets text embeddings, inserts projected image embeddings, adjusts labels and attention mask, and calls the underlying LM with `inputs_embeds`.

- [ ] **Step 4: Run focused test**

Run: `pytest tests/test_multimodal_splice.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llava_mini/model/llava_qwen.py tests/test_multimodal_splice.py
git commit -m "feat: add multimodal qwen wrapper"
```

## Task 6: Sample Trace Visualization

**Files:**
- Create: `src/llava_mini/visualize/__init__.py`
- Create: `src/llava_mini/visualize/sample_trace.py`
- Test: `tests/test_sample_trace.py`

- [ ] **Step 1: Write failing trace export test**

```python
from llava_mini.visualize.sample_trace import render_trace_html


def test_render_trace_html_contains_key_sections():
    html = render_trace_html({
        "image": "toy.png",
        "prompt_tokens": ["<image>", "What"],
        "answer_tokens": ["square"],
        "labels": [-100, -100, 123],
        "shapes": {"projected": [1, 4, 896]},
    })
    assert "label masking" in html.lower()
    assert "projected" in html
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_sample_trace.py -v`

Expected: FAIL because visualization module does not exist.

- [ ] **Step 3: Implement JSON and HTML trace rendering**

Render a compact local HTML file showing image path, prompt tokens, answer tokens, label mask, and tensor shapes.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_sample_trace.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llava_mini/visualize tests/test_sample_trace.py
git commit -m "feat: add sample trace visualization"
```

## Task 7: One-Batch Overfit Training

**Files:**
- Create: `src/llava_mini/train/__init__.py`
- Create: `src/llava_mini/train/overfit_one_batch.py`
- Create: `src/llava_mini/train/train_projector.py`
- Create: `src/llava_mini/train/train_full.py`
- Create: `scripts/run_overfit.sh`
- Test: `tests/test_training_smoke.py`

- [ ] **Step 1: Write tiny optimizer smoke test**

Keep the test synthetic so CI does not download CLIP/Qwen.

```python
import torch


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
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_training_smoke.py -v`

Expected: PASS after adding the test.

- [ ] **Step 3: Implement overfit script**

Load config, toy dataset, tokenizer, image processor, CLIP, projector, Qwen wrapper, then train for a small number of steps while writing `outputs/overfit/metrics.jsonl`.

- [ ] **Step 4: Run static and unit tests**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 5: Run real overfit command**

Run: `bash scripts/run_overfit.sh`

Expected: metrics file is created and loss decreases across the run. If model downloads are required, note first-run download time.

- [ ] **Step 6: Commit**

```bash
git add src/llava_mini/train scripts/run_overfit.sh tests/test_training_smoke.py
git commit -m "feat: add one batch overfit training"
```

## Task 8: Final Verification And Korean Handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-06-llava-mini-lab-design.md` if implementation details changed.

- [ ] **Step 1: Update README quickstart**

Document:

- environment setup
- toy data generation
- sample trace export
- one-batch overfit
- projector-only training
- full fine-tuning training
- what to inspect in each output

- [ ] **Step 2: Run full verification**

Run:

```bash
pytest -v
python scripts/prepare_toy_data.py --out data/toy
python -m llava_mini.visualize.sample_trace --config configs/qwen05b_clip.yaml --index 0 --out outputs/traces/sample0.html
bash scripts/run_overfit.sh
```

Expected:

- tests pass
- toy dataset exists
- sample trace HTML exists
- overfit metrics exist
- loss decreases

- [ ] **Step 3: Commit final docs**

```bash
git add README.md docs/superpowers/specs/2026-06-06-llava-mini-lab-design.md
git commit -m "docs: add llava mini lab quickstart"
```

- [ ] **Step 4: Report results in Korean**

Include exact commands run, output paths, and any caveats about GPU memory, model downloads, or training time.
