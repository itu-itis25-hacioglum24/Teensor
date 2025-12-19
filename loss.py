from beegrad import tensor

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

class MAE(Loss):
    """
    Mean Absolute Error
    """
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return abs(diff).mean()
    
class Huber(Loss):
    """
    Huber Loss
    """
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        
        diff = y_pred - y_true
        
        abs_diff = abs(diff)
        #Masking
        is_small_error = abs_diff <= self.delta
        
        quadratic_part = 0.5 * (diff ** 2)
        
        linear_part = self.delta * (abs_diff - 0.5 * self.delta)
        
        loss_tensor = (is_small_error * quadratic_part) + ((1 - is_small_error) * linear_part)
        
        return loss_tensor.mean()
    
    