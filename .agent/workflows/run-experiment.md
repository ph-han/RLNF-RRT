---
description: 코드 수정 후 서버에 배포하고 학습 후 결과를 가져오는 전체 연구 워크플로우
---

# Research Pipeline Workflow

로컬에서 코드 수정 → GitHub 푸시 → 서버에서 학습 → 결과 가져오기

## 전체 흐름

```
[로컬] 코드 수정 → git push → [서버] git pull → 학습 실행 → [로컬] rsync 결과 가져오기
```

---

## Step 1: 이전 결과 백업 (버전 관리)

실험 전에 이전 결과를 버전별로 백업:

```bash
# result/checkpoints 폴더가 있으면 old/로 이동
cd /home/phan/Desktop/research/RLNF-RRT/result
DATE=$(date +%Y%m%d_%H%M%S)
VERSION="v1"  # 사용자가 지정
mkdir -p old/${VERSION}_${DATE}
mv checkpoints old/${VERSION}_${DATE}/ 2>/dev/null || true
mv logs old/${VERSION}_${DATE}/ 2>/dev/null || true
```

---

## Step 2: 코드 수정 및 GitHub 푸시

// turbo
```bash
git status --short
```

변경사항 확인 후 커밋 메시지 생성:

```bash
git add .
git commit -m "<conventional commit message>"
git push origin main
```

---

## Step 3: GPU 확인 및 학습 실행

### 3.1 서버 GPU 사용 여부 확인

// turbo
```bash
sshpass -p 'phan' ssh phan@166.104.224.180 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
```

예시 출력:
```
index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, 10234 MiB, 24576 MiB, 87 %   ← 사용 중
1, 0 MiB, 24576 MiB, 0 %        ← 빈 GPU!
```

> [!IMPORTANT]
> **GPU 사용 규칙**: 선배님과 서버를 공유하므로, 모든 GPU가 사용 중이면 로컬에서 학습합니다.

---

### 3.2a 빈 GPU가 있으면 → 서버에서 학습

```bash
# GPU 1이 비어있을 경우
sshpass -p 'phan' ssh phan@166.104.224.180 << 'EOF'
cd ~/RLNF-RRT
git pull origin main
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/train_flow.py > train.log 2>&1 &
echo "Training started on GPU 1. PID: $!"
EOF
```

**학습 스크립트**: `scripts/train_flow.py`
**저장 경로**: 
- 체크포인트: `result/checkpoints/best_model.pt`
- 로그: `train.log`

---

### 3.2b 빈 GPU가 없으면 → 로컬에서 학습 🆕

> [!NOTE]
> 서버 GPU가 모두 사용 중이므로 로컬 PC에서 학습을 진행합니다.

// turbo
```bash
cd /home/phan/Desktop/research/RLNF-RRT
# CUDA (Linux) 또는 CPU 사용
uv run python scripts/train_flow.py
```

로컬 학습 시 참고:
- GPU가 없으면 자동으로 CPU 사용 (느리지만 가능)
- 학습 시간이 오래 걸릴 수 있으므로 작은 배치로 테스트 권장
- 기본 설정: epochs=50, batch_size=64

---

## Step 4: 학습 진행 상황 모니터링 (선택)

```bash
sshpass -p 'phan' ssh phan@166.104.224.180 "tail -f ~/RLNF-RRT/train.log"
```

---

## Step 5: 결과 가져오기 (rsync)

학습 완료 후 결과를 로컬로 동기화:

// turbo
```bash
# 체크포인트 가져오기
rsync -avP -e "sshpass -p 'phan' ssh -p 22" phan@166.104.224.180:~/RLNF-RRT/result/checkpoints/ /home/phan/Desktop/research/RLNF-RRT/result/checkpoints/

# 학습 로그 가져오기 (optional)
rsync -avP -e "sshpass -p 'phan' ssh -p 22" phan@166.104.224.180:~/RLNF-RRT/train.log /home/phan/Desktop/research/RLNF-RRT/result/
```

**가져올 파일들**:
- `result/checkpoints/best_model.pt` - Best validation loss 모델
- `train.log` - 전체 학습 로그

---

## Step 6: 시각화 생성 (로컬)

학습 완료 후 결과를 시각화:

// turbo
```bash
cd /home/phan/Desktop/research/RLNF-RRT

# Loss curve 생성
uv run python scripts/visualize_loss.py

# Sampling 시각화
uv run python scripts/visualize_sampling.py --num_samples 512

# Step-by-step 변환 시각화
uv run python scripts/visualize_each_step.py --num_examples 3
```

**생성되는 파일들**:
- `result/visualization/loss/loss_curve_{timestamp}.png`
- `result/visualization/sampling/samples_grid_{timestamp}.png`
- `result/visualization/sampling/sample_density_{timestamp}.png`
- `result/visualization/each_step/flow_steps_{idx}_{timestamp}.png`

---

## Step 7: Notion에 실험 결과 기록

`/research-log` 워크플로우를 호출하여:
- 실험 설정 (하이퍼파라미터: epochs, batch_size, lr, num_blocks, cond_dim)
- 결과 요약 (최종 train/val loss)
- 시각화 결과 분석
- 다음 실험 방향 추천

---

## 예시 사용

```
사용자: /run-experiment
AI: 실험을 시작합니다.

1️⃣ 이전 결과 백업: result/old/v1_20260209_140500/
2️⃣ 코드 푸시: feat(training): add Flow model training pipeline
3️⃣ 서버 업데이트: git pull 완료
4️⃣ 학습 시작: PID 12345
   → 학습이 완료되면 알려주세요!

---
사용자: 학습 끝났어
AI: 결과를 가져옵니다.

5️⃣ rsync 완료: best_model.pt (6.8 MB)

6️⃣ 시각화 생성 중...
   ✅ Loss curve 저장됨
   ✅ Sampling 시각화 저장됨
   ✅ Step-by-step 시각화 저장됨

7️⃣ 최종 결과:
   - Train Loss: 2.3456
   - Val Loss: 2.4567
   
Notion에 기록할까요?
```

---

## 참고: 현재 설정

### 학습 스크립트
- **파일**: `scripts/train_flow.py`
- **기본 설정**: epochs=50, batch_size=64, lr=1e-4

### 저장 경로
- **체크포인트**: `result/checkpoints/best_model.pt`
- **로그**: `train.log` (서버) 또는 stdout (로컬)
- **백업**: `result/old/v{N}_{timestamp}/`
- **시각화**: 
  - `result/visualization/loss/` - Loss curves
  - `result/visualization/sampling/` - Sample distributions
  - `result/visualization/each_step/` - Step-by-step transformations

### 서버 환경
- Host: `phan@166.104.224.180`
- Remote Dir: `~/RLNF-RRT`
- Dataset: 서버에 저장됨 (`data/train/`, `data/valid/`)

### 시각화 스크립트
- `scripts/visualize_loss.py` - Loss curve 생성
- `scripts/visualize_sampling.py` - Sampling 분포 시각화
- `scripts/visualize_each_step.py` - Flow 변환 단계별 시각화