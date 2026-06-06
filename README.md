# LLaVA Mini Lab

작은 LLaVA 1 스타일 모델을 직접 만들어보는 학습용 프로젝트입니다.

기본 목표는 성능보다 관찰입니다. 이미지가 CLIP feature가 되고, projector를 지나 Qwen 임베딩 공간으로 들어가고, 텍스트 토큰 사이에 삽입되며, 답변 토큰에만 loss가 걸리는 과정을 코드와 시각화로 확인합니다.

## 기본 모델

- Vision tower: `openai/clip-vit-large-patch14-336`
- LLM: `Qwen/Qwen2.5-0.5B-Instruct`
- Bridge: `VisionProjector`, 기본은 linear projection
- 기본 학습 감각: 작은 모델 full-parameter 튜닝으로 구조를 먼저 이해하고, 나중에 7B LoRA로 확장

## 설치

```bash
uv sync --extra dev
```

## 빠른 확인

```bash
uv run --extra dev pytest -v
uv run python scripts/prepare_toy_data.py --out data/toy
uv run python -m llava_mini.visualize.sample_trace --demo --out outputs/traces/demo.html
bash scripts/run_overfit.sh
```

`outputs/traces/demo.html`을 열면 다음 흐름을 볼 수 있습니다.

- prompt token
- answer token
- label masking
- CLIP patch feature shape
- projector output shape
- image embedding이 삽입된 뒤의 sequence shape

`scripts/run_overfit.sh`는 기본적으로 synthetic sanity loop를 실행합니다. 빠르게 loss 기록 형식과 학습 루프를 확인하기 위한 명령입니다.

## 실제 Qwen/CLIP one-batch overfit

먼저 toy data를 만듭니다.

```bash
uv run python scripts/prepare_toy_data.py --out data/toy
```

그다음 실제 모델을 다운로드하고 한 batch overfit을 돌립니다.

```bash
uv run python -m llava_mini.train.overfit_one_batch \
  --config configs/qwen05b_clip.yaml \
  --out outputs/real-overfit \
  --real
```

첫 실행은 Hugging Face 모델 다운로드 때문에 오래 걸릴 수 있습니다. A5000 8장 환경에서는 이후 `accelerate launch`나 DDP/LoRA 경로를 추가해서 확장하면 됩니다.

## 코드에서 먼저 볼 부분

- `src/llava_mini/data/collator.py`: prompt token, answer token, label masking, image token 위치
- `src/llava_mini/model/projector.py`: CLIP feature를 LLM hidden size로 바꾸는 projector
- `src/llava_mini/model/llava_qwen.py`: text embedding 안의 `<image>` 자리를 image patch embedding들로 치환
- `src/llava_mini/train/overfit_one_batch.py`: synthetic sanity loop와 실제 Qwen/CLIP one-batch overfit
- `src/llava_mini/visualize/sample_trace.py`: 학습 샘플 trace HTML 렌더링

## 현재 범위와 다음 확장

현재 버전은 구조 학습용 최소 구현입니다. 다음 단계에서 추가하기 좋은 것은 다음과 같습니다.

- 실제 dataset trace 생성: tokenizer와 CLIP processor 결과를 한 샘플별 JSON/HTML로 저장
- projector-only training script 분리
- `accelerate` config와 multi-GPU 실행 명령
- Qwen full fine-tune와 LoRA/QLoRA config 분리
- loss curve plot과 generated answer snapshot dashboard
```
