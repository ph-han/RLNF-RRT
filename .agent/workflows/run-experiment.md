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
# result/images 폴더가 있으면 old/로 이동
cd /home/phan/Desktop/research/RLNF-RRT/result
DATE=$(date +%Y%m%d_%H%M%S)
VERSION="v8"  # 사용자가 지정
mkdir -p old/${VERSION}_${DATE}
mv images old/${VERSION}_${DATE}/ 2>/dev/null || true
mv models old/${VERSION}_${DATE}/ 2>/dev/null || true
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

## Step 3: 서버 GPU 확인 및 학습 실행

먼저 빈 GPU를 확인:

```bash
sshpass -p 'phan' ssh phan@166.104.224.180 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
```

예시 출력:
```
index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, 10234 MiB, 24576 MiB, 87 %   ← 사용 중
1, 0 MiB, 24576 MiB, 0 %        ← 빈 GPU!
```

빈 GPU를 찾아서 학습 실행:

```bash
# GPU 1이 비어있을 경우
sshpass -p 'phan' ssh phan@166.104.224.180 << 'EOF'
cd ~/RLNF-RRT
git pull origin main
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_plannerflows.py > train.log 2>&1 &
echo "Training started on GPU 1. PID: $!"
EOF
```

또는 사용자가 직접 서버에서 학습을 실행할 경우:

```bash
ssh phan@166.104.224.180
# 비번: phan
nvidia-smi  # GPU 확인
cd ~/RLNF-RRT && git pull
CUDA_VISIBLE_DEVICES=<빈_GPU_번호> python scripts/train_plannerflows.py
```

---

## Step 4: 학습 진행 상황 모니터링 (선택)

```bash
ssh phan@166.104.224.180 "tail -f ~/RLNF-RRT/train.log"
```

---

## Step 5: 결과 가져오기 (rsync)

학습 완료 후 결과를 로컬로 동기화:

// turbo
```bash
rsync -avP -e "ssh -p 22" phan@166.104.224.180:~/RLNF-RRT/result/images /home/phan/Desktop/research/RLNF-RRT/result/
rsync -avP -e "ssh -p 22" phan@166.104.224.180:~/RLNF-RRT/result/models /home/phan/Desktop/research/RLNF-RRT/result/
```

---

## Step 6: Notion에 실험 결과 기록

`/research-log` 워크플로우를 호출하여:
- 실험 설정 (하이퍼파라미터 등)
- 결과 요약 (최종 loss 등)
- 다음 실험 방향 추천

---

## 예시 사용

```
사용자: /run-experiment
AI: 실험을 시작합니다.

1️⃣ 이전 결과 백업: result/old/v7_20260206_135500/
2️⃣ 코드 푸시: feat(encoder): add ResNet18 backbone
3️⃣ 서버 업데이트: git pull 완료
4️⃣ 학습 시작: PID 12345
   → 학습이 완료되면 알려주세요!

---
사용자: 학습 끝났어
AI: 결과를 가져옵니다.

5️⃣ rsync 완료: 78개 이미지, 2개 모델
6️⃣ 최종 결과:
   - loss_curve_v7_ep780.png (최종 loss: -0.6648)
   
Notion에 기록할까요?
```

---

## 참고: 서버 환경

- Host: `phan@166.104.224.180`
- Remote Dir: `~/RLNF-RRT`
- Dataset: 서버에 저장됨
- 결과: `result/images/`, `result/models/`
