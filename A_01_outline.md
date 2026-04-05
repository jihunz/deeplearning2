# How Backpropagation is Implemented — Building a Mini Autograd Engine from Scratch

## 1. Research Background
- Backpropagation의 두 접근: Analytical vs Computational Graph
- PyTorch가 computational graph 방식을 채택한 이유
- **동기**: 코드 레벨에서 computational graph의 구축·실행 흐름을 직접 이해하기 위해 NumPy로 autograd engine을 밑바닥부터 구현

## 2. Theoretical Development
- 2.1 Computational Graph의 정의와 구조 (node = 연산, edge = 데이터 흐름)
- 2.2 Forward Pass: 값 계산 + 그래프 동적 구축
- 2.3 Backward Pass: topological sort → 역순 chain rule
- 2.4 Local gradient 패턴 (sum→distributor, mul→swap, max→router)
- 2.5 Multiple path contribution: gradient 누적(+=)의 수학적 근거
- 2.6 PyTorch autograd 내부 구조와의 대응 (grad_fn, next_functions, AccumulateGrad)

## 3. Programming Verification
- 3.1 `Value` 클래스 설계 (data, grad, _backward, _prev)
- 3.2 연산자 구현 (+, *, **, relu, sigmoid) — 각 local gradient 정의
- 3.3 Topological sort 기반 `backward()` 구현
- 3.4 MLP 구축 (Neuron → Layer → MLP)
- 3.5 실험: XOR 학습
- 3.6 PyTorch autograd와 수치 비교 검증

## 4. Conclusion
- Computational graph 방식의 장점 (모듈성, 확장성, 자동화)
- 직접 구현에서 얻은 인사이트
- PyTorch autograd의 추가 최적화 요소

## Appendix
- A. 전체 소스코드
- B. 실험 결과 시각화 (loss curve, gradient flow)
- C. AI 활용 로그
