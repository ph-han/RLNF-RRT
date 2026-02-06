---
description: 실험 결과, 코드 변경사항 분석, 시각화 결과를 Notion에 기록합니다
---

# Research Log Workflow

실험 결과와 코드 변경사항을 분석하고, 시각화 결과 이미지와 함께 Notion에 기록합니다.

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
git diff HEAD~1 --stat
git diff HEAD~1 -- "*.py" | head -100
```

분석할 내용:
- 어떤 파일이 변경되었는지
- 주요 변경 로직은 무엇인지
- 변경의 목적이 무엇인지

---

## Step 2: 학습 결과 확인

// turbo
```bash
# Loss curve 이미지와 모델 파일 확인
ls -la result/images/ | tail -5
ls -la result/models/ | tail -5
```

---

## Step 3: 테스트 및 시각화 실행 (선택)

서버에서 테스트 스크립트 실행:

```bash
sshpass -p 'phan' ssh phan@166.104.224.180 << 'EOF'
cd ~/RLNF-RRT/scripts
source ../.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python test_plannerflows.py
# 또는
# CUDA_VISIBLE_DEVICES=0 python visualize_flow_states.py --ckpt ../result/models/latest.pth
EOF
```

결과 이미지를 로컬로 가져오기:
// turbo
```bash
rsync -avP -e "sshpass -p 'phan' ssh" phan@166.104.224.180:~/RLNF-RRT/result/visualization/ /home/phan/Desktop/research/RLNF-RRT/result/visualization/
```

---

## Step 4: 결과 이미지 랜덤 샘플링 및 분석

// turbo
```bash
# result/visualization/ 폴더에서 랜덤 10개 선택
ls result/visualization/*.png | shuf | head -10
```

선택된 이미지들을 분석:
- 샘플 분포가 적절한지
- Start/Goal 위치 표시가 올바른지
- Ground Truth와의 비교

---

## Step 5: Notion에 기록

### 5.1 코드 변경사항 기록

```
## 📝 코드 변경사항
- 변경된 파일 목록
- 주요 변경 내용 요약
- 변경 목적/의도
```

### 5.2 실험 결과 기록

```
## 📊 실험 결과
- 학습 설정 (epochs, batch_size, lr 등)
- Final Loss 값
- Loss Curve 분석
```

### 5.3 시각화 결과 분석 (10개 샘플)

```
## 🖼️ 시각화 결과 분석
각 이미지에 대해:
- 샘플 분포 품질 (좋음/보통/나쁨)
- 특이사항 또는 개선점
```

### 5.4 다음 단계 추천

```
## 🎯 다음 단계 추천
1. [추천 1]
2. [추천 2]
3. [추천 3]
```

---

## 예시 호출

```
사용자: /research-log

# AI 분석 결과
📝 **코드 변경사항**
- `MapEncoder.py`: ResNet18 backbone 추가, 3채널 입력 지원
- `train_plannerflows.py`: AdamW optimizer 적용

📊 **실험 결과**
- Epochs: 500
- Final Train Loss: -0.65
- Final Val Loss: -0.12

🖼️ **시각화 분석** (랜덤 10개)
- res_42.png: 샘플 분포 우수, free-space 커버리지 좋음
- res_128.png: 장애물 근처 샘플링 주의 필요
...

🎯 **다음 단계 추천**
1. Learning rate decay 적용 실험
2. 더 많은 epoch 학습
3. Heatmap sigma 조정 실험

✅ Notion에 기록 완료
```

---

## Notes

- 코드 변경사항은 가장 최근 커밋 기준으로 분석
- 이미지 분석은 result/visualization/ 폴더 사용
- Notion API는 외부 URL만 지원하므로 이미지는 텍스트로 분석 기록
