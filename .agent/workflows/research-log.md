---
description: 실험 결과 및 코드 변경사항을 Notion에 기록하고 다음 방향을 추천받습니다
---

# Research Log Workflow

실험 결과와 코드 변경사항을 Notion에 기록하고, AI가 다음 연구 방향을 추천합니다.

## 사용법

이 워크플로우는 다음과 같은 상황에서 호출합니다:
- 실험 결과를 기록하고 싶을 때
- 코드 변경사항을 문서화하고 싶을 때
- 다음 연구 방향에 대한 추천을 받고 싶을 때

## Steps

### 1. 현재 상태 분석

사용자에게 다음을 확인합니다:
- **기록 유형**: Experiment (실험) / Code Change (코드 변경) / Idea (아이디어)
- **제목**: 간단한 제목
- **요약**: 무엇을 했는지, 결과가 어땠는지

### 2. 변경사항 자동 감지

```bash
# 최근 git diff 확인
git diff --stat HEAD~1
git log -1 --pretty=format:"%s"
```

### 3. AI 분석 및 추천

코드 변경사항과 실험 결과를 분석하여:
- 현재 진행 상황 요약
- 잠재적 문제점 파악
- 다음 단계 추천 생성

### 4. Notion에 기록

Notion MCP를 사용하여 기록:

**옵션 A: 데이터베이스가 있는 경우**
```
mcp_notion-mcp-server_API-post-page 도구 사용:
- parent: { database_id: "<DATABASE_ID>" }
- properties: Title, Date, Type, Status, Summary, Next Steps
```

**옵션 B: 데이터베이스가 없는 경우**

Notion 검색으로 "RLNF" 관련 페이지를 찾고, 해당 페이지에 블록으로 추가:
```
mcp_notion-mcp-server_API-post-search: query="RLNF"
mcp_notion-mcp-server_API-patch-block-children: block_id=<page_id>, children=[...]
```

### 5. 결과 확인

생성된 Notion 페이지 URL을 사용자에게 공유합니다.

## 예시 호출

사용자: `/research-log`

응답 예시:
```
📝 Research Log Entry

**제목**: MapEncoder ResNet18 적용
**유형**: Code Change
**날짜**: 2026-02-06

**요약**:
- MapEncoder를 3채널 입력 (occupancy grid, start/goal heatmap) 지원으로 변경
- Pre-trained ResNet18 backbone 적용
- 첫 Conv layer와 최종 FC layer 수정

**다음 단계 추천**:
1. 학습률 조정 실험 (1e-4 → 5e-5)
2. Heatmap sigma 값 튜닝
3. Ablation study: 각 채널의 기여도 분석

✅ Notion에 기록 완료
```
