# LLaVA-1.0 (Visual Instruction Tuning) 학습 파이프라인

> 출처: Liu et al., *Visual Instruction Tuning*, NeurIPS 2023.
> 이 문서는 원조 LLaVA(=LLaVA-1.0)의 **데이터**와 **학습 단계**를 정리한 것입니다.
> 이 저장소(`llava_mini`)는 이 구조를 작은 LLM(Qwen2.5-1.5B)으로 바꿔 학습용으로 재현한 것입니다.

---

## 1. 전체 구성 (Architecture)

```
이미지 ──▶ Vision Encoder ──▶ projection W ──▶ 이미지 토큰 ┐
            (CLIP, frozen)      (linear)                    ├─▶ LLM (Vicuna) ──▶ 답변
텍스트 ────────────────────────▶ word embedding ───────────┘
```

| 구성 | LLaVA-1.0 |
|------|-----------|
| Vision encoder | CLIP **ViT-L/14** (원 논문 224px, 이후 336px 체크포인트도 공개). penultimate layer의 grid feature 사용, CLS 토큰 제외 |
| Bridge | **단일 linear projection** 행렬 `W` (Zᵥ → 언어 임베딩 공간 Hᵥ) — *LLaVA-1.5에서 2-layer MLP로 바뀜* |
| LLM | **Vicuna-13B** (7B 버전도 있음), LLaMA 기반 |

핵심 아이디어: 이미지를 CLIP feature로 만든 뒤, `W`로 **LLM의 word embedding 공간과 같은 차원의 "visual token"** 으로 바꿔, 텍스트 토큰들과 한 시퀀스에 섞어 LLM에 넣는다.

---

## 2. 데이터 (Data)

LLaVA-1.0은 두 종류의 데이터를 단계별로 쓴다.

### 2-1. Feature alignment용 — CC-595K

- **CC3M**(Conceptual Captions 3M)을 필터링해 만든 **595K image–text 쌍**.
- caption을 단순 확장(naive expansion)해 single-turn 지시 형식으로 변환:
  - 입력: `이미지` + "이미지를 간단히 설명하라" 류의 질문(여러 표현 중 랜덤)
  - 정답: 원래 caption
- 목적: 이미지 feature를 LLM 임베딩 공간에 **정렬(align)** 시키기 위한 약한 데이터.

### 2-2. Visual instruction tuning용 — LLaVA-Instruct-158K

- **COCO 이미지**를 바탕으로 **text-only GPT-4**에게 생성시킨 **158K개**의 instruction-following 데이터.
- GPT-4는 이미지를 직접 못 보므로, 이미지를 **상징적 표현(symbolic representation)** 으로 줘서 생성:
  - **captions** (장면 설명 문장들)
  - **bounding boxes** (객체 클래스 + 좌표)
  - 사람이 만든 few-shot 예시로 생성 품질을 유도(seed).
- 3가지 응답 유형으로 구성:

| 유형 | 수량 | 내용 |
|------|------|------|
| Conversation | 58K | 이미지에 대한 다중 턴 Q&A 대화 |
| Detailed description | 23K | 이미지 상세 묘사 |
| Complex reasoning | 77K | 단계적 추론이 필요한 질문/답 |
| **합계** | **158K** | |

> ScienceQA 같은 학술 벤치마크로 fine-tune하는 변형도 논문에 함께 제시됨.

### 2-3. 데이터 공개 현황 (Availability)

두 데이터 모두 **Hugging Face `liuhaotian` 계정에 공개**돼 있다. 단, **주석(JSON)** 과 **이미지**의 공개 방식이 다르다.

| 단계 | 데이터셋 | 위치 | 포함 |
|------|----------|------|------|
| Stage 1 | `LLaVA-CC3M-Pretrain-595K` | [HF](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) | `chat.json` + `metadata.json` + 필터된 이미지(`images.zip`) |
| Stage 2 | `LLaVA-Instruct-150K` | [HF](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | GPT-4 생성 instruction JSON (대화/상세/추론 split) |

- **이미지는 별도**: `LLaVA-Instruct-150K`에는 **JSON만** 있고, 실제 이미지는 **COCO `train2017`** 을 [cocodataset.org](https://cocodataset.org)에서 따로 받아야 한다. JSON에는 `image: "000000xxxxxx.jpg"` 파일명만 들어있다. (반면 CC3M-595K는 이미지까지 함께 제공.)
- **이름 vs 실제 수량**: repo 이름은 "150K"지만 학습엔 158K(대화 58K + 상세 23K + 추론 77K)를 쓴다. `conversation_58k.json`, `detail_23k.json`, `complex_reasoning_77k.json`, 합본 `llava_instruct_150k.json` 등이 들어있다.
- **라이선스 주의**: instruction 데이터는 **GPT-4로 생성** → OpenAI 약관 영향으로 **연구/비상업** 성격. 이미지는 각 원본 라이선스(COCO, CC3M)를 따름. "다운로드 가능 = 상업적 자유 사용"이 아니다.
- *(참고)* LLaVA-1.5의 정렬 데이터는 별도 [`liuhaotian/LLaVA-Pretrain`](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)(LAION-CC-SBU 558K)이며, 본 문서의 1.0 데이터와 다르다.

---

## 3. 학습 단계 (Two-Stage Training)

LLaVA-1.0은 **2단계**로 학습한다. 공통적으로 **vision encoder(CLIP)는 항상 frozen**.

### Stage 1 — Pre-training for Feature Alignment

- **데이터**: CC-595K (2-1)
- **freeze**: Vision encoder ❄️ + LLM ❄️
- **학습 대상**: **projection 행렬 `W` 만** ✅
- 목적: 이미지 feature ↔ LLM word embedding 정렬. 사실상 "호환되는 visual tokenizer"를 학습.
- 대표 하이퍼파라미터: 1 epoch, lr `2e-3`, batch size 128, cosine schedule.

### Stage 2 — Fine-tuning End-to-End

- **데이터**: LLaVA-Instruct-158K (2-2) — *멀티모달 챗봇용*. (또는 ScienceQA)
- **freeze**: Vision encoder ❄️
- **학습 대상**: **projection `W` + LLM(Vicuna) 가중치** ✅✅
- 목적: 시각 지시(visual instruction)를 따르는 대화/추론 능력 학습.
- 대표 하이퍼파라미터: 3 epochs, lr `2e-5`, batch size 32.

```
            Vision(CLIP)     Projection W      LLM(Vicuna)
Stage 1        ❄️ frozen        ✅ train         ❄️ frozen
Stage 2        ❄️ frozen        ✅ train         ✅ train
```

---

## 4. 입력 시퀀스 구성과 Loss masking

멀티턴 대화를 하나의 토큰 시퀀스로 직렬화하고, **assistant(정답) 토큰에만 loss**를 건다.

```
[System message] <STOP>
Human: Xinstruct¹ <STOP> Assistant: Xanswer¹ <STOP>
Human: Xinstruct² <STOP> Assistant: Xanswer² <STOP> ...
```

- 첫 턴의 instruction은 `[이미지, 질문]` 또는 `[질문, 이미지]` 순서를 랜덤 배치.
- 이미지는 projection을 거친 visual token들로 시퀀스에 삽입.
- **학습 목표**: auto-regressive language modeling loss.
- **마스킹**: system message·human 발화 토큰은 loss에서 제외(-100), **Assistant 답변 토큰(+ 정지 토큰)만** loss 계산.

---

## 5. 이 저장소(`llava_mini`)와의 대응

| LLaVA-1.0 | 이 repo |
|-----------|---------|
| CLIP ViT-L/14 (penultimate, CLS 제외) | `model/vision_tower.py` — `hidden_states[-2][:, 1:, :]` |
| linear projection `W` | `model/projector.py` — `nn.Linear` |
| Vicuna-13B | **Qwen2.5-1.5B-Instruct** 로 교체 (`model/llava_qwen.py`) |
| visual token splice | `model/llava_qwen.py` — `splice_image_embeddings` |
| answer-only loss masking | `data/collator.py` — `build_labels` |
| Stage 1 (projector-only) | `train/train_projector.py` (LLM·vision freeze) |
| Stage 2 (projector + LLM) | `train/train_full.py` (vision freeze) |

> 차이: 본 repo는 학습 데이터로 CC-595K / Instruct-158K 대신 **toy 데이터 한 batch overfit**으로 구조와 데이터 흐름 검증에 초점을 둔다. 프롬프트 템플릿도 Vicuna 대화체 대신 Qwen **ChatML**(`<|im_start|>`)을 쓴다.
```
