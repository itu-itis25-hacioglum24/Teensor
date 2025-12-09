from beegrad import tensor

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """
        reset grad values to zero
        """
        for p in self.parameters:
            p.grad.fill(0) 

    def step(self):
        """
        Update parameters using gradient descent
        """
        for p in self.parameters:
            # p.data  -= learning_rate * p.grad 
            p.data -= self.lr * p.grad
