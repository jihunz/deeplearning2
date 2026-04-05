from Value import Value
from nn import MLP

if __name__ == '__main__':
    # ── 1. XOR 데이터 ──
    xs = [
        [Value(0), Value(0)],
        [Value(0), Value(1)],
        [Value(1), Value(0)],
        [Value(1), Value(1)],
    ]
    ys = [0, 1, 1, 0]

    # ── 2. 모델 생성: 입력2 → 은닉4 → 출력1 ──
    model = MLP(2, [8, 4, 1])

    # ── 3. 학습 루프 ──
    lr = 0.05
    epochs = 1000

    for epoch in range(epochs):
        # ① forward
        preds = [model(x) for x in xs]

        # ② loss: MSE = Σ(pred - target)²
        loss = Value(0)
        for pred, y in zip(preds, ys):
            loss = loss + (pred - y) ** 2

        # ③ zero grad (backward 전에!)
        for p in model.parameters():
            p.grad = 0.0

        # ④ backward
        loss.backward()

        # ⑤ gradient descent
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | loss = {loss.data:.6f}")

    # ── 4. 최종 결과 ──
    print("\n=== XOR 결과 ===")
    for x, y in zip(xs, ys):
        pred = model(x)
        print(f"input={[xi.data for xi in x]} → pred={pred.data:.4f}, target={y}")