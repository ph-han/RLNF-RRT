---
description: 실험 결과, 코드 변경사항 분석, 시각화 결과를 Notion에 기록합니다
---

# Research Log Workflow

> [!IMPORTANT]
> **Notion Template V2**: Notion 기록 시 반드시 **Template**을 적용해야 합니다.
> - **Code Changes**: 코드의 단순 나열이 아닌 **"개선점"**과 **"결과에 미친 영향"**을 위주로 분석합니다.
> - **Properties**: `Type`, `Status`, `Date` 속성을 정확히 태깅해야 템플릿이 정상 동작합니다.

실험 결과와 코드 변경사항을 분석하고, Notion에 심층 기록합니다.

## 사용법

이 워크플로우는 다음과 같은 상황에서 호출합니다:
- 실험 결과를 기록하고 싶을 때
- 코드 변경사항을 문서화하고 싶을 때
- 결과 이미지를 분석하고 기록하고 싶을 때

---

## Step 1: 코드 변경사항 분석

// turbo
```bash
# 최근 커밋과의 diff 확인
# 최근 커밋과의 diff 확인 (상세)
git diff HEAD~1 --stat
git show HEAD --format=format:%B
git diff HEAD~1  # 전체 diff 확인 (너무 길면 주요 파일만 선택)
```

분석할 내용 (단순 나열 금지 🚫):
- **Core Logic**: 어떤 알고리즘이나 로직이 변경되었는가?
- **Intent**: 왜 이 변경이 필요했는가? (개선점, 버그 수정 등)
- **Impact**: 이 변경이 실험 결과에 어떤 영향을 미칠 것으로 예상되는가?

---

## Step 2: 학습 결과 확인

// turbo
```bash
# Checkpoint 확인
ls -lh result/checkpoints/ | tail -5

# Visualization 결과 확인
ls -la result/visualization/loss/*.png 2>/dev/null | tail -3
ls -la result/visualization/sampling/*.png 2>/dev/null | tail -3
ls -la result/visualization/each_step/*.png 2>/dev/null | tail -3
```

---

## Step 3: 테스트 및 시각화 실행 (로컬)

> [!NOTE]
> 학습은 서버에서, 테스트/시각화는 로컬에서 실행합니다.

### 3.1 서버에서 모델 파일 가져오기

// turbo
```bash
# 서버에서 최신 모델 파일 다운로드
rsync -avP -e "sshpass -p 'phan' ssh" phan@166.104.224.180:~/RLNF-RRT/result/checkpoints/ /home/phan/Desktop/research/RLNF-RRT/result/checkpoints/
```

### 3.2 로컬에서 테스트 실행

// turbo
```bash
cd /home/phan/Desktop/research/RLNF-RRT
# Loss curve 생성
uv run python scripts/visualize_loss.py

# Sampling 시각화
uv run python scripts/visualize_sampling.py --num_samples 512

# Step-by-step 시각화
uv run python scripts/visualize_each_step.py --num_examples 3
```

---

## Step 4: 결과 이미지 랜덤 샘플링 및 분석

### 4.1 Loss Curve 분석
// turbo
```bash
# Loss curve 시각화 스크립트 실행
cd /home/phan/Desktop/research/RLNF-RRT
uv run python scripts/visualize_loss.py

# 생성된 loss curve 확인
ls result/visualization/loss/*.png | tail -5
```

분석 포인트:
- Loss가 수렴하고 있는지
- 학습 안정성 (급격한 변동 없는지)
- Best validation loss 지점

### 4.2 Sampling 결과 분석
// turbo
```bash
# Sampling 시각화 스크립트 실행
cd /home/phan/Desktop/research/RLNF-RRT
uv run python scripts/visualize_sampling.py --num_samples 512

# 생성된 sampling 결과 확인
ls result/visualization/sampling/*.png | tail -5
```

분석 포인트:
- 샘플 분포가 ground truth와 유사한지
- Start/Goal 위치 표시가 올바른지
- 장애물 회피가 잘 되는지

### 4.3 Each Step 분석
// turbo
```bash
# Step-by-step 변환 시각화 실행
cd /home/phan/Desktop/research/RLNF-RRT
uv run python scripts/visualize_each_step.py --num_examples 3 --num_samples 300

# 생성된 each_step 결과 확인
ls result/visualization/each_step/*.png | tail -5
```

분석 포인트:
- 단계별 flow 변화가 부드러운지 (z0 → z1 → ... → x)
- Base distribution에서 최종 경로로의 변환 과정
- Ground Truth와의 비교

---

## Step 4.5: Compare with Paper (논문 비교 분석)

`reference/` 폴더의 논문을 기반으로 현재 구현과 비교 분석합니다.

### 확인 사항

// turbo
```bash
# reference 폴더 내 논문 목록 확인
ls -la reference/*.pdf 2>/dev/null || echo "No PDF files in reference/"
```

### 분석 포인트

다음 관점에서 현재 코드가 논문과 일치하는지 검토:

1. **Methodology 일치성**
   - 논문의 핵심 알고리즘이 제대로 구현되었는가?
   - 수식이나 pseudo-code와 코드 로직이 일치하는가?

2. **Architecture 적합성**
   - 모델 구조가 논문에서 제안한 것과 동일한가?
   - 레이어 구성, 차원 설정 등이 논문과 맞는가?

3. **실험 설정**
   - Hyperparameter (learning rate, batch size 등)가 논문과 유사한가?
   - 데이터셋 구성이 논문의 실험 환경과 비슷한가?

4. **Gap 분석**
   - 논문에 있지만 아직 구현되지 않은 부분은?
   - 논문과 다르게 구현한 부분이 있다면 그 이유는?

### Notion 기록 포맷 (추가 섹션)

```
## 5. 논문 비교 (Compare with Paper)
- **참조 논문**: [논문 제목]
- **일치하는 부분**: [내용]
- **차이점/Gap**: [내용]
- **다음 구현 목표**: [내용]
```

> [!TIP]
> 자세한 논문 비교 분석은 `/paper-review` workflow를 사용하세요.
> 분석 결과는 `reference/REVIEW_NOTES.md`에 저장됩니다.

---

## Step 5: Notion에 기록

### 5.2 Notion 기록 구조 (한글)

다음 **5가지** 섹션으로 구성하여 기록합니다:

1. **핵심 변화** (Key Changes)
   - 무엇이 바뀌었는지 한 문장으로 요약
   - 예: "Epoch 수를 100에서 10으로 줄여 파이프라인 검증 속도 향상"

2. **분석** (Analysis)
   - 왜 바꾸었는지(의도), 어떤 로직이 핵심인지 설명
   - 예: "전체 학습을 기다리지 않고 아티팩트 생성 및 전송 기능만 빠르게 테스트하기 위함."

3. **상세한 실험 결과 분석 결과 (코드 분석 포함)**
   - 실험 수치(Loss, Epoch 등)와 코드 레벨의 변경 사항 매핑
   - Diff 내용 분석 포함

4. **다음으로 해보면 좋을 거 추천**
   - 이번 실험을 바탕으로 제안할 점
   - 예: "파이프라인이 검증되었으니 실제 학습(Epoch 500) 시작 권장"

5. **논문 비교** (Compare with Paper) 🆕
   - 참조 논문과 현재 구현의 일치도 분석
   - Gap 분석 및 다음 구현 목표

```
## 1. 핵심 변화
[내용]

## 2. 분석
[내용]

## 3. 상세한 실험 결과 분석 결과 (코드 분석 포함)
[내용]
```diff
[Diff]
```

## 4. 다음으로 해보면 좋을 거 추천
[내용]

## 5. 논문 비교 (Compare with Paper)
- **참조 논문**: [논문 제목]
- **일치하는 부분**: [현재 잘 구현된 내용]
- **차이점/Gap**: [아직 구현되지 않은 부분]
- **다음 구현 목표**: [우선순위가 높은 구현 항목]
```

---

## 예시 호출

```
사용자: /research-log

# AI 분석 결과 (Notion 기록용)

## 1. 핵심 변화
`train_plannerflows.py`의 `num_epochs`를 100에서 10으로 수정하여 실험 주기를 단축함.

## 2. 분석
- **의도**: MCP 워크플로우(`/run-experiment`)가 정상 작동하는지 빠르게 검증하기 위함.
- **영향**: 모델 수렴은 불가능하지만, 로그 저장 및 결과 전송 파이프라인 테스트에는 충분함.

## 3. 상세한 실험 결과 분석 결과 (코드 분석 포함)
- **실험 데이터**:
  - Final Loss: -0.2864 (수렴하지 않음, 정상)
  - Epoch: 10/10 완료
- **코드 변경점**:
  - `scripts/train_plannerflows.py`: 학습 루프 반복 횟수 수정
```diff
- num_epochs = 100
+ num_epochs = 10
```

## 4. 다음으로 해보면 좋을 거 추천
1. 파이프라인 검증이 완료되었으므로 `num_epochs`를 다시 500 이상으로 복구하고 실제 학습 수행.
2. `vis_debug.py`를 실행하여 서버 모델의 레이어 구조 불일치 문제 해결 시도.


✅ Notion에 기록 완료
```

---

## Notes

- 코드 변경사항은 가장 최근 커밋 기준으로 분석
- 이미지 분석은 result/visualization/ 폴더 사용
- Notion API는 외부 URL만 지원하므로 이미지는 텍스트로 분석 기록