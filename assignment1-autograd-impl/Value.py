from math import exp


class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._prev = set()
        self._backward = lambda: None  # 기본: 아무것도 안 함 (leaf node)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # int/float 자동 변환
        out = Value(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            # add의 local gradient = 1, 1 (distributor)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # 2 + Value 지원
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            # mul의 local gradient = swap (상대방 값)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # 2 * Value 지원
        return self * other

    def __neg__(self):  # -Value
        return self * -1

    def __sub__(self, other):  # Value - other
        return self + (-other)

    def __rsub__(self, other):  # other - Value
        return other + (-self)

    def backward(self):
        # 1. topological sort
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # 2. 자기 자신(loss)의 grad = 1
        self.grad = 1.0

        # 3. 역순으로 _backward 호출
        for node in reversed(topo):
            node._backward()

    def __pow__(self, n):  # n은 상수 (int/float)
        out = Value(self.data ** n)  # forward: self.data를 n제곱
        out._prev = {self}  # 단항 연산이니 자기 자신

        def _backward():
            self.grad += (n * self.data ** (n - 1)) * out.grad  # 멱함수 미분: n * x^(n-1)

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data))  # forward: max(0, self.data)
        out._prev = {self}

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad  # 양수면 1, 아니면 0

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + exp(-self.data))  # forward: 1 / (1 + exp(-self.data))
        out = Value(s)
        out._prev = {self}

        def _backward():
            self.grad += s * (1 - s) * out.grad  # sigmoid 미분: s * (1 - s)

        out._backward = _backward
        return out
