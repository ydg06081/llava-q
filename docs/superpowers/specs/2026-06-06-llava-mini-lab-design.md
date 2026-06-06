# LLaVA 1 Mini Lab Design

## Goal

Build a small, inspectable LLaVA 1 style training project that teaches how visual instruction tuning works by exposing the model structure, data transformations, token insertion, label masking, training loss, and generated outputs.

The first runnable model will use:

- Vision encoder: CLIP vision tower, frozen by default.
- Language model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Bridge: trainable linear projector from CLIP patch features into the Qwen hidden size.

The project should run real training on the user's A5000 x 8 machine while still supporting tiny single-batch experiments for learning and debugging.

## Learning Priorities

The project is optimized for understanding, not leaderboard performance.

The user should be able to inspect:

- How a raw image becomes CLIP pixel values.
- How CLIP outputs patch embeddings.
- How the projector maps image features into the language model embedding space.
- Where image embeddings are inserted into the text sequence.
- How prompts and answers are tokenized.
- Which labels are masked with `-100`.
- Which answer tokens contribute to loss.
- How loss changes during projector training and full fine-tuning.
- How a model overfits one tiny batch.

## Recommended Training Path

Start with the small full-fine-tuning track, then leave an upgrade path for 7B LoRA experiments.

1. One-sample and one-batch tracing.
2. Projector-only overfit run.
3. Projector plus Qwen full-parameter fine-tuning on a tiny dataset.
4. Optional LoRA path later for larger models.

This keeps the early learning loop visible and fast while preserving a config structure that can later point at larger LLaMA/Vicuna-style or Qwen models.

## Architecture

The model will mirror the LLaVA 1 idea:

```text
image
  -> CLIP image processor
  -> CLIP vision encoder
  -> patch features
  -> linear projector
  -> image embeddings in Qwen hidden size

prompt + answer
  -> Qwen tokenizer
  -> text token embeddings
  -> splice projected image embeddings at the <image> placeholder
  -> causal LM forward pass
  -> loss on answer tokens only
```

The CLIP vision tower is frozen by default. The projector is always trainable. The Qwen language model can be configured as frozen, full fine-tuned, or later LoRA-tuned.

## Components

### Model

- `vision_tower.py`: Loads CLIP image processor and vision model, returns patch features.
- `projector.py`: Defines the linear or small MLP projector.
- `llava_qwen.py`: Combines text embeddings and projected image embeddings, builds attention masks, forwards through Qwen, and generates answers.

### Data

- `dataset.py`: Reads JSONL records with `image`, `question`, and `answer`.
- `collator.py`: Builds padded batches, inserts image placeholder metadata, and creates masked labels.
- `transforms_debug.py`: Produces serializable traces for a single sample.

### Training

- `overfit_one_batch.py`: Minimal sanity training loop to prove the model can learn one batch.
- `train_projector.py`: Projector-only training mode.
- `train_full.py`: Projector plus full Qwen fine-tuning mode.

### Visualization

- `sample_trace.py`: Exports an HTML/JSON trace of one sample's image preprocessing, tokenization, image-token insertion, and label mask.
- `training_dashboard.py`: Produces local training plots and sample predictions.

### Configuration

- `qwen05b_clip.yaml`: Default config for Qwen2.5-0.5B plus CLIP.

## Dataset Format

Use JSONL for the first version:

```json
{"image": "data/toy/images/cat.jpg", "question": "What is in this image?", "answer": "A cat is sitting on a blanket."}
```

The training prompt template should include a literal `<image>` placeholder. The collator will replace that placeholder with projected image embeddings during the forward pass while keeping labels masked for the prompt and image positions.

## Error Handling

- Missing image files should raise a clear path error.
- Records without `question` or `answer` should fail validation before training.
- Samples without `<image>` in the rendered prompt should raise a clear prompt-template error.
- Shape mismatches between CLIP features, projector output, and Qwen hidden size should include the expected and actual dimensions.

## Testing

The first test suite should cover behavior that is easy to break:

- JSONL dataset loading and validation.
- Prompt rendering with exactly one `<image>` placeholder.
- Label masking so only answer tokens contribute to loss.
- Projector output shape.
- Model forward pass with synthetic image features.
- One-batch overfit smoke test with tiny settings where possible.

## Success Criteria

The first complete version is successful when:

1. A toy dataset can be generated locally.
2. A single sample trace can be exported and inspected.
3. A one-batch overfit run decreases loss.
4. The model can generate a short answer after training.
5. The config can switch between projector-only and full fine-tuning.
6. The code is organized so a later 7B LoRA track can be added without rewriting the data path.
