import numpy as np


class Teensor():
    
    def __init__(self,data,children=(),_op=''):
        
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32  )
        else:
            self.data = np.array(data, dtype=np.float32)
            
        self._backward = lambda:None
        self.op = _op
        self.prev = children
        self.grad = np.zeros_like(self.data,dtype=np.float32)
        
    def __repr__(self):
        return f'Teensor(Data:{self.data},dtype:Teensor)'

    def _unbroadcast(self, grad, original_shape):
        # 1. sum the extra dimensions (Örn: (32, 10) -> (10,))
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)
        
        # 2. sum the dimensions that were originally 1 (Örn: (1, 10) -> (10,))
        for i, dim in enumerate(original_shape):
            if dim == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # ---- BASIC OPERATIONS -----

    def __add__(self,other):
        
        other = other if isinstance(other,Teensor) else Teensor(other)
        
        
        out = Teensor(self.data + other.data, (self,other));out.op='+'
            
        ### (d(A+B)/dA = I[identity matrix])I  * out.grad (chain rule)
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += other._unbroadcast(out.grad, other.data.shape)
                
        out._backward = _backward
            
            
        return out
        
        
        
    def __mul__(self,other): # This operation is the Hadamard product, not matrix multiplication
        
        other = other if isinstance(other,Teensor) else Teensor(other)

        
        out = Teensor(self.data * other.data, (self,other));out.op = '*'
            
        ### (d(A*B)/dA = B) B * out.grad(chain rule) 
        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += other._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
                
        return out

    def __pow__(self,other): # This works via the Hadamard product , not matrix multiplication
        
        assert isinstance(other,(int,float)), "Only int and float are allowed for exponent"
        
        out = Teensor(self.data**other, (self, ));out.op = 'pow'
            
        def _backward():
            self.grad += self._unbroadcast((other * self.data**(other-1)) * out.grad, self.data.shape)                
        out._backward = _backward
            
        return out
    
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    
        
    # ---- LINEAR  ALGEBRA OPERATIONS ----
    
    @property
    def transpose(self):
        out = Teensor(self.data.transpose(), (self,), 'T')        # gradient of A^t = out.grad^t
        def _backward():
            self.grad += out.grad.transpose()
        
        out._backward = _backward
            
        return out
    
    
    # Matrix Multiplication A@B
    def matmul(self,other):
        
        other = other if isinstance(other, Teensor) else Teensor(other)
        out = Teensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += self._unbroadcast(out.grad @ other.data.swapaxes(-1, -2), self.data.shape)
            # dL/dB = A.T @ dL/dY
            other.grad += other._unbroadcast(self.data.swapaxes(-1, -2) @ out.grad, other.data.shape)
        
        out._backward = _backward
        
        return out


    @property
    def inverse(self):
        out = Teensor(np.linalg.inv(self.data),(self, ),'inverse')
        
        def _backward():
            # upstream gradient
            G = out.grad
            self.grad += -out.data @ G @ out.data  # matmul
        out._backward = _backward
        
        return out
    
    def sum(self, axis=None, keepdims=False):
        
        data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Teensor(data, (self,), 'sum')

        def _backward():
            grad = out.grad

            if axis is None:
                self.grad += grad * np.ones_like(self.data)

            else:
                expanded_grad = grad
                if not keepdims:
                    if isinstance(axis, int):
                        expanded_grad = np.expand_dims(grad, axis)
                    else:
                        for ax in sorted(axis):
                            expanded_grad = np.expand_dims(expanded_grad, ax)

                self.grad += expanded_grad * np.ones_like(self.data)

        out._backward = _backward
        
        return out

    # Compute gradients for all tensors
    def backward(self, grad=None):
        if grad is None:
            # generate gradient of 1 for scalar outputs
            if self.data.shape == () or self.data.size == 1:
                self.grad = np.ones_like(self.data)
            else:
                # if the output is not a scalar, grad must be provided
                raise RuntimeError("Grad must be specified for non-scalar outputs")
        else:
            self.grad = grad

        visited = set()
        topo = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def mean(self):
        # mean of the Teensor elements -skaler result
        div = self.data.size # number of elements
        out = Teensor(np.mean(self.data), (self,), 'mean')
        
        def _backward():
            # d(mean)/dx = 1/N
            # divide the upstream gradient by number of elements
            self.grad += (1.0 / div) * out.grad
            
        out._backward = _backward
        return out