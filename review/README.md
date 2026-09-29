# Furiosa-AI-Training 복습 자료

이 디렉터리는 저장소의 학습 내용을 바탕으로 만든 복습 자료와 문제를 관리합니다.

## 운영 방식

- 기준 브랜치: `main`
- 매일 20:00(KST) Telegram으로 Markdown 복습 자료를 전송합니다.
- 평일: 당일 학습·변경 내용 중심의 요약과 문제
- 토·일요일: 주간 복습 문제
- 월말 평가: 평일에는 보내지 않고, 해당 월의 마지막 주말에 월간 평가를 보냅니다.
- 문제 답변은 Telegram 대화에서 분석하고, 이해가 부족한 주제와 다음 복습 계획을 `progress.md` 및 날짜별 답변 기록에 저장합니다.
- 새 복습 자료는 생성 후 `main`에 직접 commit/push합니다.

## 자료 구성

- `YYYY-MM-DD-initial-review.md`: 전체 범위 최초 복습 자료
- `YYYY-MM-DD-daily-review.md`: 평일 복습
- `YYYY-MM-DD-weekly-review.md`: 주간 복습
- `YYYY-MM-DD-monthly-review.md`: 월간 평가
- `answers/`: Telegram 답변 분석 기록
- `progress.md`: 누적 취약 주제와 학습 진도
