"""
우리 Value autograd vs 수학적 해석해 (analytical gradient) 수치 비교 검증
동일한 입력, 동일한 연산 → gradient가 일치해야 함
"""
from math import exp
from Value import Value


def check(name, ours, expected, tol=1e-6):
    ok = abs(ours - expected) < tol
    print(f"  {name}: ours={ours:.6f}, expected={expected:.6f} {'✓' if ok else '✗'}")
    return ok


all_pass = True

# ════════════════════════════════════════
print("검증 1: 기본 연산 (add, mul, pow)")
print("  e = (a * (a + b)) ** 2,  a=2, b=3")
print("  e = (2 * 5)^2 = 100")
print("  ∂e/∂a = 2*(a*(a+b)) * (2a+b) = 2*10*7 = 140")
print("  ∂e/∂b = 2*(a*(a+b)) * a      = 2*10*2 = 40")
# ── 우리 autograd ──
a = Value(2.0); b = Value(3.0)
e = (a * (a + b)) ** 2
e.backward()

all_pass &= check("e.data", e.data, 100.0)
all_pass &= check("a.grad", a.grad, 140.0)
all_pass &= check("b.grad", b.grad, 40.0)

# ════════════════════════════════════════
print("\n검증 2: ReLU")
print("  w = relu(x + y) * 2,  x=-3, y=4")
print("  w = relu(1) * 2 = 2")
print("  ∂w/∂x = 2 * 1 (relu pass) = 2")
print("  ∂w/∂y = 2 * 1 (relu pass) = 2")
x = Value(-3.0); y = Value(4.0)
w = (x + y).relu() * Value(2.0)
w.backward()

all_pass &= check("w.data", w.data, 2.0)
all_pass &= check("x.grad", x.grad, 2.0)
all_pass &= check("y.grad", y.grad, 2.0)

# ════════════════════════════════════════
print("\n검증 2b: ReLU (음수 입력 → gradient 차단)")
print("  w = relu(x + y) * 2,  x=-5, y=1")
print("  w = relu(-4) * 2 = 0")
print("  ∂w/∂x = 0 (relu blocked)")
x2 = Value(-5.0); y2 = Value(1.0)
w2 = (x2 + y2).relu() * Value(2.0)
w2.backward()

all_pass &= check("w2.data", w2.data, 0.0)
all_pass &= check("x2.grad", x2.grad, 0.0)
all_pass &= check("y2.grad", y2.grad, 0.0)

# ════════════════════════════════════════
print("\n검증 3: Sigmoid")
print("  c = sigmoid(a) * 3,  a=1.5")
s = 1 / (1 + exp(-1.5))
print(f"  sigmoid(1.5) = {s:.6f}")
print(f"  c = {s*3:.6f}")
print(f"  ∂c/∂a = 3 * s * (1-s) = {3*s*(1-s):.6f}")
a2 = Value(1.5)
c2 = a2.sigmoid() * Value(3.0)
c2.backward()

all_pass &= check("c2.data", c2.data, s * 3)
all_pass &= check("a2.grad", a2.grad, 3 * s * (1 - s))

# ════════════════════════════════════════
print("\n검증 4: Multiple path contribution")
print("  f = a*a + a*b,  a=3, b=5")
print("  f = 9 + 15 = 24")
print("  ∂f/∂a = 2a + b = 11")
print("  ∂f/∂b = a = 3")
a3 = Value(3.0); b3 = Value(5.0)
f = a3 * a3 + a3 * b3
f.backward()

all_pass &= check("f.data", f.data, 24.0)
all_pass &= check("a3.grad", a3.grad, 11.0)
all_pass &= check("b3.grad", b3.grad, 3.0)

# ════════════════════════════════════════
print("\n" + "=" * 50)
print(f"전체 결과: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
print("=" * 50)
