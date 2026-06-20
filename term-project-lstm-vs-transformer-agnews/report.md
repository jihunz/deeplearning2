# LSTM vs Transformer Encoder for Text Classification (AG News)

Team: 장지훈 (ETRI School, 02521122) · Deep Learning 2, 2026

## 1. Introduction
본 프로젝트는 AG News 4-class 뉴스 주제 분류에서 recurrent 구조인 LSTM과 self-attention 구조인 Transformer Encoder를 동일한 조건으로 비교한다. 두 모델은 모두 from scratch로 학습한다. research question은 "데이터, preprocessing, vocabulary, sequence length, 학습 예산을 동일하게 통제했을 때 두 구조가 성능, 수렴, data efficiency, 오분류 양상에서 어떻게 다른가"이다. 목표는 높은 정확도 자체가 아니라 차이의 원인을 설명하는 것이다. LSTM과 Transformer는 sequence encoding에서 가장 널리 쓰인 두 구조이며, 동일 과제에서의 통제된 직접 비교는 각 구조의 강점과 약점을 드러낸다.

## 2. Dataset
출처는 HuggingFace의 AG News single source이며, 4개 class(0 World, 1 Sports, 2 Business, 3 Sci/Tech)로 구성된다. test set은 최종 평가에만 사용하고, validation set은 training set의 10%를 분리해 구성했다(seed 42).

| Split | 샘플 수 |
|---|---|
| Train | 108,000 |
| Validation | 12,000 |
| Test | 7,600 |

| Class | Train | Test |
|---|---|---|
| World | 26,991 | 1,900 |
| Sports | 26,966 | 1,900 |
| Business | 27,100 | 1,900 |
| Sci/Tech | 26,943 | 1,900 |

Preprocessing은 두 모델에 동일하게 적용했다. 단어 단위 tokenization(소문자화) 후, vocabulary는 training set에서만 구축했다(상한 20,000, minimum frequency 2, padding token과 unknown token 포함). 각 문장은 최대 길이 128로 자르거나 padding했고, padding 위치는 masking으로 무시했다. training set의 고유 단어는 67,575종이며, 빈도 상위 20,000종으로 전체 token 출현의 97.3%를 덮는다. 따라서 unknown token으로 처리되는 비율은 약 2.7%로 작다.

Label indexing은 흔한 오류 지점이다. raw CSV 버전은 label이 1에서 4이지만 HuggingFace 버전은 0에서 3이라 cross-entropy loss가 요구하는 0-based index와 바로 호환된다. single source만 사용해 이 혼선을 피했다.

## 3. Models
두 모델의 공유 골격은 동일하다. embedding(차원 128) → encoder → masked mean pooling → linear layer(출력 4) 순서이며, loss는 cross-entropy를 사용한다. 두 모델은 가운데 encoder만 다르다. 이 통제가 비교의 핵심이며, 성능 차이를 구조 차이로 귀속할 수 있게 한다.

| 구성 | LSTM | Transformer Encoder |
|---|---|---|
| Embedding dimension | 128 | 128 |
| Encoder | bidirectional LSTM, 2 layers, hidden size 128 | 2 layers, 4 attention heads, feed-forward network dimension 256, sinusoidal positional encoding |
| Output dimension | 256 (bidirectional) | 128 (model dimension) |
| Pooling / Loss | masked mean pooling / cross-entropy | masked mean pooling / cross-entropy |
| Trainable parameters | 3,220,484 | 2,825,476 |

embedding(약 256만 parameters)이 양쪽에서 공통으로 지배적이므로, 총 parameter 규모는 약 14% 차이로 비교 가능한 수준이다.

Implementation 세부는 다음과 같다. padding token은 index 0으로 고정하고 embedding의 padding index로 지정해 그 vector가 학습되지 않게 했다. pooling과 self-attention은 padding 위치를 masking으로 무시한다. 두 모델 모두 동일한 masked mean pooling을 사용해, pooling 방식이 비교의 교란 요인이 되지 않도록 했다.

## 4. Experiments
학습 설정은 Adam optimizer, learning rate 0.001, batch size 64, 최대 8 epochs, dropout 0.1, seed 42이다. model selection은 validation 성능 기준 best checkpoint로만 수행했고, test set은 최종 1회만 평가했다. 학습 환경은 Mac MPS이다.

Main comparison은 위 설정으로 두 모델을 학습한다. Ablation은 data efficiency를 측정하기 위해 training set을 5%, 25%, 50%, 100%로 줄여 두 모델을 학습한다. 이때 vocabulary는 전체 training set에서 한 번 구축해 모든 비율에 고정 재사용하고, 부분 추출은 class 비율을 유지(stratified)하며 seed를 고정한다. validation set과 test set은 항상 전체를 사용한다.

평가 지표는 accuracy, macro F1-score, training/validation loss curve, confusion matrix이다. 공정 비교를 위해 split, tokenizer, vocabulary, 최대 길이, pooling, optimizer, learning rate, batch size, epoch 수, seed를 두 모델에 동일하게 고정했다. 유일한 변수는 encoder 구조다.

## 5. Results

| Model | Test accuracy | Test macro F1-score |
|---|---|---|
| LSTM | 0.833 | 0.835 |
| Transformer | 0.910 | 0.909 |

Transformer가 약 7.7 percentage point 앞선다. 비슷한 parameter 규모에서의 차이다.

[[IMG:fig_loss_curves.png]]
Figure 1. Training and validation loss curves (left: LSTM, right: Transformer).

학습 곡선이 성능 차이의 직접 증거다. LSTM은 training loss가 0.035까지 감소하는 동안 validation loss가 0.54에서 약 2.0으로 증가한다. 전형적인 overfitting이다. Transformer는 validation loss가 약 0.28에서 0.33으로 평탄하게 유지되어 overfitting이 거의 없고, 첫 epoch에 이미 88% 정확도로 빠르게 수렴한다. validation 기준 best checkpoint가 overfitting 이전 시점(epoch 2)을 저장해 LSTM의 최종 성능을 확보했다.

[[IMG:fig_confusion.png]]
Figure 2. Confusion matrices (row: true label, column: predicted label).

혼동 양상은 두 모델에서 다음과 같이 나타난다. LSTM에서는 Business가 주로 Sci/Tech로 오분류되고(258건), Sci/Tech도 주로 Business로 오분류되어(154건) 두 class의 상호 혼동이 가장 크다. World와 Sports는 오류 분포가 서로 비슷하며, 둘 다 주로 Business와 Sci/Tech로 오분류된다. 결과적으로 LSTM은 모든 class에서 상당한 오류를 내어 오류가 전반에 퍼져 있다. Transformer에서는 오류가 거의 Business와 Sci/Tech의 경계에 집중되고(Business를 Sci/Tech로 147건, Sci/Tech를 Business로 138건), Sports는 거의 완벽하며(오분류 38건) World도 깨끗하다.

공통점은 두 모델 모두 Business와 Sci/Tech를 가장 많이 혼동한다는 것이다. 두 class가 기업, 기술, 시장 관련 어휘를 공유하기 때문이며, sequence length가 아니라 의미 중첩이 혼동의 원인이다. class별로 보면 Transformer는 Sports와 World에서 특히 강하고 Business와 Sci/Tech에서만 약하다. LSTM은 Sports와 World에서도 Transformer만큼 강하지 않아, 어휘가 뚜렷한 class에서조차 오류가 적지 않다.

[[IMG:fig_ablation.png]]
Figure 3. Data-efficiency ablation. Test accuracy versus training set size.

| Training data | LSTM | Transformer | Gap |
|---|---|---|---|
| 5% (5,399) | 0.727 | 0.798 | +0.071 |
| 25% (26,998) | 0.846 | 0.875 | +0.029 |
| 50% (53,999) | 0.833 | 0.897 | +0.064 |
| 100% (108,000) | 0.833 | 0.910 | +0.077 |

Transformer는 데이터가 늘수록 정확도가 단조롭게 향상된다. LSTM은 25%에서 0.846으로 정점을 기록한 뒤 정체하며, 데이터를 더 늘려도 overfitting 때문에 향상되지 않는다. 즉 LSTM은 약 25% 지점에서 성능이 saturation되어 추가 데이터를 활용하지 못하고, Transformer는 데이터에 따라 안정적으로 scaling된다. Transformer가 모든 구간에서 앞서므로, Transformer가 LSTM보다 더 많은 학습 데이터를 요구한다는 사전 가설(data efficiency가 더 낮다는 가설)은 본 과제에서 기각되었다. AG News는 비교적 쉬운 과제라 5,400개 규모에서도 Transformer가 충분히 일반화했다.

학습 효율에서도 차이가 있었다. Mac MPS에서 epoch당 학습 시간은 Transformer가 약 47초로 LSTM의 약 56초보다 짧았다. self-attention은 위치 간 연산을 병렬화할 수 있는 반면 LSTM은 token을 순차적으로 처리하기 때문이다. 따라서 Transformer는 정확도와 학습 속도 양쪽에서 앞섰다.

종합하면, 두 모델의 성능 차이는 sequence order 처리 능력(다음 절에서 보듯 이 과제는 order를 거의 활용하지 않는다)이 아니라 generalization과 overfitting의 차이에서 비롯한다.

## 6. Failure Analysis

| 본문 요약 | 정답 | LSTM 예측 | Transformer 예측 | 추정 원인 |
|---|---|---|---|---|
| Some People Not Eligible to Get in on Google IPO | Sci/Tech | Business | Business | 기업의 IPO라 금융과 기술의 경계가 모호함 |
| Intel to delay product aimed for high-definition TVs | Business | Sci/Tech | Sci/Tech | 어휘는 기술이지만 label은 사업 영향, 두 모델 모두 높은 확신으로 오분류 |
| Teenage T. rex's monster growth | Sci/Tech | Business | Sci/Tech | 명백한 과학 기사인데 LSTM만 낮은 확신으로 오분류 |
| IBM to hire even more new workers | Sci/Tech | Business | Sci/Tech | 고용 기사라 경계가 모호, Transformer만 정확히 분류 |
| Prediction Unit Helps Forecast Wildfires | Sci/Tech | Sci/Tech | Sports | Transformer의 드문 오분류 |

카테고리별 개수는 두 모델 모두 오분류 428건, LSTM만 오분류 841건, Transformer만 오분류 259건이다. LSTM 단독 오류가 약 3.2배 많아 정확도 차이를 뒷받침한다. 두 모델이 함께 오분류한 사례의 61%는 정답이 Business 또는 Sci/Tech다. 이들은 label 자체가 모호한 경계 사례로, 모델의 결함이라기보다 과제의 class 중첩에서 비롯한다.

확신도 양상도 다르다. 공통 오분류는 종종 높은 확신을 동반한다. 예를 들어 Intel 기사에서는 두 모델 모두 0.98 이상의 확신으로 오분류했다. 모호한 경계에서의 overconfidence다. 반면 LSTM 단독 오류는 0.50에서 0.59 수준의 낮은 확신을 동반하는 경우가 많다. 모델이 스스로도 불확실한 상태에서 오분류한 것으로, robustness 부족을 시사한다. 요약하면 두 모델은 서로 다른 양상으로 실패한다.

## 7. Extension: Transformer Mechanism Analysis
계획한 세 가지 메커니즘 분석을 모두 수행했다. positional encoding 실험만 추가 학습이 필요하고, 나머지는 학습된 모델을 그대로 사용한다.

첫째, positional encoding on/off 실험이다. positional encoding을 제거한 Transformer를 동일 조건으로 다시 학습한 결과, test accuracy는 0.910에서 0.911로 사실상 변하지 않았다(차이 -0.001). self-attention은 positional encoding이 없으면 순서 정보가 없는 set encoder가 된다. 그럼에도 정확도가 동일하다는 것은, AG News가 단어 순서보다 어휘 출현이 결정적인 bag-of-words 성격임을 의미한다. 이 과제에서 Transformer의 순서 주입 부품은 불필요하다.

둘째와 셋째는 gradient saliency와 attention weight를 통한 token importance 분석이다. 동일한 Sports 기사에 대해 두 모델이 어느 token을 근거로 판단하는지 비교했다.

[[IMG:fig_mechanism.png]]
Figure 4. Token importance on a Sports example (true label: Sports). Left: LSTM gradient saliency. Center and right: Transformer gradient saliency and attention weight.

LSTM은 주제와 무관한 단어 "giddy"에 가장 큰 importance를 부여하고 이 기사를 Sci/Tech로 오분류했다. 반면 Transformer는 "phelps", "seconds", "medley", "400" 같은 주제 token에 importance를 분산했고, gradient saliency와 attention weight가 같은 경향을 보이며 기사를 정확히 Sports로 분류했다. 즉 Transformer는 판단 근거를 여러 주제 token에 분산하는 반면, LSTM은 특정 token 하나에 과도하게 의존할 수 있다. 이는 정확도 차이의 token 수준 원인을 보여준다. 다만 gradient saliency와 attention weight가 완전히 일치하지는 않으므로, attention weight를 곧바로 설명으로 등치할 수는 없다.

## 8. Conclusion
- 동일 조건에서 Transformer가 LSTM보다 분명히 우수하다(test accuracy 0.910 대 0.833).
- 성능 차이의 원인은 generalization 능력의 차이다. LSTM은 심하게 overfitting하고 약 25% 데이터 지점에서 saturation되어 추가 데이터를 활용하지 못한다. Transformer는 데이터에 따라 안정적으로 scaling된다.
- 이 과제는 단어 순서가 거의 중요하지 않은 bag-of-words 성격이다. positional encoding 제거가 정확도에 영향을 주지 않았다. 따라서 성능 차이는 순서 처리 능력이 아니라 robustness와 generalization에서 비롯한다.
- Transformer가 더 많은 학습 데이터를 요구한다는 사전 가설은 기각되었다. 모든 데이터 구간에서 Transformer가 앞섰다.
- 한계와 향후 과제로, 결과는 단일 seed 기준이므로 여러 seed로 재확인이 필요하다. LSTM의 dropout 등 regularization을 강화하면 성능 차이가 줄어드는지 검증할 가치가 있다. 또한 단어 순서가 중요한 과제(sentiment analysis, natural language inference)에서 positional encoding의 효과를 재검증하고, 단어 순서를 무작위로 섞는 추가 ablation으로 두 구조의 순서 의존성을 더 직접 비교할 수 있다.
