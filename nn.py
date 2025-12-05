import numpy as np
from Teensor.Tensor import Teensor
    

class Module:
    @property
    def parameters(self):
        params = []
        #collect parameters from submodules
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters)
        return params   
    def forward(self, x):
        return x
    
    def __call__(self, x):
        # when get called, call forward method
        return self.forward(x)

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Teensor(np.random.randn(in_features, out_features))
        self.bias = Teensor(np.random.randn(1, out_features))

    def forward(self, x):
        return x.matmul(self.weight) + self.bias
    
    @property
    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        out = Teensor(np.maximum(0, x.data), (x,), 'ReLU')

        def _backward():
            x.grad += (x.data > 0) * out.grad

        out._backward = _backward
        return out