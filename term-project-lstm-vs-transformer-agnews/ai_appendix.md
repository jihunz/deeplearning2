# AI Usage Appendix

Team: 장지훈 (ETRI School, 02521122) · Deep Learning 2, 2026

## 1. 사용한 AI 도구
Claude (Anthropic, Claude Code 환경)를 coding 보조 겸 학습 tutor로 사용했다.

## 2. AI를 어디에 사용했나
- 개념 설명: LSTM의 recurrence와 gate, self-attention과 multi-head attention, positional encoding, masking, overfitting, data efficiency 등.
- 코드 초안: preprocessing pipeline, 두 모델, training과 evaluation loop, ablation, gradient saliency와 attention 분석, 그림 생성.
- debugging, 결과 해석 보조, 계획서와 보고서 초안 작성.
- 진행은 단순 대필이 아니라, 결정 지점마다 예측, 실행, 피드백, 회상을 반복하는 tutor 방식으로 개념을 체득하는 데 중점을 두었다.

## 3. 부정확하거나 불완전했던 AI 출력
- background 학습을 nohup과 백그라운드 연산자로 이중 실행해, 완료 알림이 실제 학습이 아니라 실행 shell의 종료를 가리킨 오류.
- 통계 script의 format string 오류로 실행이 실패한 경우.
- 보고서 PDF에 처음 선택한 한글 글꼴이 PDF 라이브러리와 호환되지 않아 다른 TrueType 글꼴로 교체.
- mechanism 분석 예시를 실제 test 기사 대신 손으로 줄여 입력해 예측이 달라진 오류. 실제 기사로 교체해 바로잡았다.

## 4. AI 출력에서 팀이 수정하거나 보완한 것
- dropout을 임의의 비대칭 값(0.3과 0.1)에서, 공통 0.1로 시작한 뒤 validation 기준으로 조정하도록 통제했다.
- extension 주제를 AI가 처음 제안한 active learning 연계에서 Transformer mechanism 분석으로 변경했다.
- 계획서와 보고서의 문체를 간결하고 정확하게 다듬고 불필요한 표현을 제거했다.

## 5. 팀이 내린 결정
- ablation 주제(data efficiency)의 선택, 그리고 더 통찰적인 대안을 찾도록 요구한 것.
- extension의 범위와 방향, 제출물 구성과 문체.
- 각 실험 전 가설과 예측 수립(예: 데이터를 줄이면 어느 모델이 더 크게 떨어질지), 그리고 결과 해석.
- 팀 구성과 일정.

## 6. AI 사용이 프로젝트에 미친 영향
구현 속도를 높여 개념 이해와 결과 해석에 시간을 집중할 수 있었다. 예측과 피드백을 반복하는 문답이 개념 체득에 도움이 되었다. 동시에 AI 출력에는 위와 같은 오류가 있어 모든 코드, 결과, 주장을 직접 검증해야 했다. 최종 책임은 팀에 있다.
