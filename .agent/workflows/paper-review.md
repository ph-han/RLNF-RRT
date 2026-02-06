---
description: Reference 논문 기반 코드 리뷰 및 조언 제공
---

# Paper-Based Code Review Workflow

이 workflow는 `reference/` 폴더의 논문 PDF를 기반으로 현재 코드를 리뷰하고 조언을 제공합니다.

## 사용법

`/paper-review` 명령어를 사용하거나, "논문 기반으로 코드 리뷰해줘" 라고 요청하면 됩니다.

## Steps

### 1. Reference 폴더 확인
- `reference/` 폴더 내 PDF 파일 목록 확인
- 새로운 논문이 있는지 체크

### 2. 논문 분석
- PDF 파일을 읽고 핵심 methodology, algorithm, architecture 파악
- 주요 contribution과 implementation detail 정리

### 3. 현재 코드 분석
- `src/` 폴더의 현재 구현 상태 파악
- 논문의 방법론과 비교

### 4. 리뷰 및 조언 제공
다음 항목들을 체크하고 피드백 제공:
- [ ] **Methodology 일치성**: 논문의 핵심 알고리즘이 제대로 구현되었는가?
- [ ] **Architecture 적합성**: 모델/시스템 구조가 논문과 일치하는가?
- [ ] **Missing components**: 논문에 있지만 구현되지 않은 부분은?
- [ ] **Implementation quality**: 코드 품질, 최적화, 모범 사례 준수
- [ ] **방향성 조언**: 다음 단계로 무엇을 해야 하는가?

### 5. 결과 정리
- `reference/REVIEW_NOTES.md`에 리뷰 결과 정리 (optional)
- 우선순위가 높은 개선점 하이라이트

## 참고사항
- `reference/` 폴더는 `.gitignore`에 추가되어 GitHub에 업로드되지 않습니다
- 논문 PDF는 저작권을 고려하여 개인 참고용으로만 사용하세요
