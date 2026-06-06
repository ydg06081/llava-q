# LLaVA Mini Lab

작은 LLaVA 1 스타일 모델을 직접 만들어보는 학습용 프로젝트입니다.

기본 목표는 성능보다 관찰입니다. 이미지가 CLIP feature가 되고, projector를 지나 Qwen 임베딩 공간으로 들어가고, 텍스트 토큰 사이에 삽입되며, 답변 토큰에만 loss가 걸리는 과정을 코드와 시각화로 확인합니다.

## 첫 실행 흐름

```bash
uv run --extra dev pytest -v
uv run python scripts/prepare_toy_data.py --out data/toy
```
