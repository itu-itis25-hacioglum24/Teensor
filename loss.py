from Teensor.Tensor import Teensor

class Loss:

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Not implemented lil boi.")

class MSE(Loss):
    """
    Mean Squared Error 
    """
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return (diff ** 2).mean()

