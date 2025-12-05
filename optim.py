from Teensor.Tensor import Teensor

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """
        reset grad values to zero
        """
        for p in self.parameters:
            # param.grad bir numpy array olduğu için direkt fill() çalışır.
            p.grad.fill(0) 

    def step(self):
        """
        Update parameters using gradient descent
        """
        for p in self.parameters:
            # p.data (Teensor'un verisi) -= learning_rate * p.grad (Gradyan Array'i)
            p.data -= self.lr * p.grad