from random import uniform
from Value import Value


class Neuron:
    def __init__(self, nin, activation=True):
        self.w = [Value(uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        ws = self.b
        for wi, xi in zip(self.w, x):
            ws = ws + wi * xi
        return ws.relu() if self.activation else ws

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, activation=True):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs  # 출력 1개면 리스트 벗기기

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):  # nouts = [4, 2, 1]
        sz = [nin] + nouts           # [3, 4, 2, 1]
        self.layers = [Layer(sz[i], sz[i+1], activation=(i != len(nouts)-1))
                       for i in range(len(nouts))]  # 마지막 레이어는 activation 없음

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)             # 이전 출력 = 다음 입력
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
