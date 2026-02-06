---
description: 코드 변경사항을 자동으로 커밋하고 GitHub에 푸시합니다
---

# Git Push Workflow

코드 변경 시 의미있는 커밋 메시지를 자동 생성하고 GitHub에 푸시합니다.

## 사용법

코드 수정 후 이 워크플로우를 호출하면:
1. 변경된 파일을 자동 감지
2. 변경 내용을 분석하여 커밋 메시지 생성
3. 스테이징, 커밋, 푸시를 한 번에 수행

## Steps

### 1. 변경사항 확인

// turbo
```bash
git status --short
```

변경된 파일이 없으면 사용자에게 알리고 종료합니다.

### 2. 변경 내용 분석

// turbo
```bash
git diff --staged
git diff
```

변경된 코드를 분석하여 다음을 파악합니다:
- 어떤 파일이 변경되었는지
- 주요 변경 내용이 무엇인지
- 변경의 목적이 무엇인지

### 3. 커밋 메시지 생성

Conventional Commits 형식으로 커밋 메시지 생성:

```
<type>(<scope>): <subject>

<body>
```

**Type 종류**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `docs`: 문서 변경
- `style`: 코드 스타일 변경
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 등

### 4. 스테이징 및 커밋

사용자에게 생성된 커밋 메시지를 보여주고 확인 후:

```bash
git add .
git commit -m "<generated commit message>"
```

### 5. GitHub 푸시

```bash
git push origin <current-branch>
```

### 6. 결과 확인

푸시 성공 시 GitHub 커밋 URL을 제공합니다.

## 예시 호출

사용자: `/git-push`

응답 예시:
```
📊 Git Status:
 M src/rlnf_rrt/models/map_encoder.py
 M src/rlnf_rrt/training/trainer.py

📝 생성된 커밋 메시지:
feat(encoder): replace CNN with pretrained ResNet18

- Update MapEncoder to accept 3-channel input
- Add pretrained ResNet18 as backbone
- Modify first conv layer for 3-channel input
- Adjust final FC layer for latent dimension output

이대로 커밋하시겠습니까? (Y/n)

✅ 커밋 완료: abc1234
✅ GitHub 푸시 완료
🔗 https://github.com/ph-han/RLNF-RRT/commit/abc1234
```

## Notes

- 커밋 전 항상 변경 내용을 사용자에게 보여줍니다
- 민감한 파일(.env, secrets 등)은 자동으로 제외합니다
- 푸시 실패 시 원인을 분석하고 해결 방법을 제안합니다
