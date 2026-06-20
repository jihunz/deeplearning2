# Initial Plan

## 1. Team
- Team: 장지훈 (ETRI 스쿨, 02521122)
- Team Leader: 장지훈 (1인 팀)

## 2. Dataset Split
- HuggingFace `ag_news` 단일 소스, 4-class (라벨 0–3, 변환 불필요).
- `train_test_split(test_size=0.1, seed=42)` → train 108k / val 12k / test 7.6k. 클래스 균등.
- test셋은 최종 평가에만 사용 (모델 선택·튜닝·early stopping 금지).

## 3. Preprocessing
- 토큰화: 단어 단위 (소문자화 + 구두점 분리).
- 어휘(vocabulary): train에서만 구축, 최대 20k, 최소빈도 2, PAD·UNK 토큰.
- 최대 길이 128 (초과는 자르고, 부족하면 패딩), 패딩은 마스킹 처리.
- 두 모델에 동일한 입력 사용.

## 4. Model Configuration
| | LSTM | Transformer Encoder |
|---|---|---|
| 임베딩 | 128 | 128 (= d_model) |
| 본체(core) | BiLSTM 2층, hidden 128 | 2층 · 어텐션 헤드 4 · FFN 256 · 위치 인코딩(sinusoidal) |
| 풀링 | 평균 (패딩 마스킹) | 평균 (패딩 마스킹) |
| 출력 / 손실 | Linear→4 · Cross-Entropy | Linear→4 · Cross-Entropy |

- 공통: Adam, 학습률 1e-3, 배치 64, 최대 8 epoch, dropout 0.1 (공통 시작 → val로 조정), seed 42, 모델 선택은 val로만.
- 파라미터 수: LSTM ≈ 3.22M / Transformer ≈ 2.83M (비슷한 규모). 사전학습 모델 금지.

## 5. Ablation Plan
- 주제: 데이터 효율 — train 5 / 25 / 50 / 100% (≈ 5k / 27k / 54k / 108k).
- 가설: 데이터가 적으면 Transformer가 불리(귀납 편향이 약함), LSTM은 소량에서도 안정적. 데이터가 늘수록 둘 격차 감소.
- 통제: 어휘는 전체 train으로 한 번 만들어 고정·재사용, 부분 추출은 클래스 비율 유지(stratified) + 고정 seed, val/test는 항상 전체.
- 그림: x = 데이터량 / y = 정확도·macro-F1, 두 모델을 한 그래프에. 100% 지점은 메인 비교로 재사용.
- 추가(선택): Transformer 내부 분석 — 위치 인코딩 on/off · 어텐션 가중치 시각화 · gradient saliency(입력 기여도).

## 6. Expected Analysis
- 성능: 둘 다 정확도 ≈ 90%+, Transformer가 근소 우위이거나 비슷.
- 데이터 효율: 5%에서 LSTM ≥ Transformer, 100%에서 비슷해짐 (가설 검증).
- 수렴: Transformer는 빠르지만 학습률에 민감, LSTM은 안정적 (loss curve로 확인).
- 혼동(confusion): Business ↔ Sci/Tech가 가장 많을 것. 오분류 ≥ 5건 분석(공통 / LSTM만 / Transformer만): 모호한 표현·짧은 문장·잘림(truncation).
- 메커니즘: 위치 인코딩을 꺼도 정확도 유지 예상 = AG News는 단어 순서보다 어휘 중심(bag-of-words). 어텐션·saliency는 클래스 키워드에 집중할 것.

---
환경: Mac MPS / Colab T4, seed 42, 체크포인트 → Drive. 산출물: Jupyter notebook + 보고서(4–6p) + AI Usage Appendix(≤1p).
