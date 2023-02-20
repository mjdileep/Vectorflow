import numpy as np


class Layer:
    """
        The base class of the Layer 
    """
    def __call__(self, x, **kwargs):
        return self.forward(x)

    def grad_zero():
        """
            Set accumilated gradients to zero in the layer 
        """
        for each in self.grads.keys():
            self.grads[each]=np.abs(self.grads[each]*0.0)

class Linear(Layer):
    """
       Linear layer class 
    """
    
    def __init__(self, in_size, out_size):
        """
            Constructor of Liniear Layer 
            Inputs:
            - in_size: input dimention size of the data (d)
            - out_size: output dimention size of the data (D)
            - weight_init_fn: Weight initialization fn
        """
        self.params = {}
        self.grads = {}
        self.params["w"] = np.random.randn(in_size, out_size)* np.sqrt(2/(in_size**2+out_size**2))
        self.params["b"] = np.zeros(out_size)
        self.cache = None
        self.grads["w"] = np.zeros((in_size, out_size))
        self.grads["b"] = np.zeros(out_size)
        
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, d_1, ... d_k)

            Returns:
            - y: Output data, of shape (N, out_1, ... out_k)
        """
        out = x.dot(self.params["w"])+self.params["b"]
        self.cache = x
        return out
    
    def backward(self, dx):
        """
            Computes the backward pass for an linear layer.

            Inputs:
            - dx: Upstream gradient data, of shape (N, out_1, ... out_k)
        """
        x = self.cache
        self.grads["w"] += x.T.dot(dx)
        self.grads["b"] += np.sum(dx, axis=0)
        dout = dx.dot(self.w.T)
        return dout

class ReLU(Linear):
    """
       ReLU layer class 
    """
    
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, d_1, ... d_k)

            Returns:
            - y: Output data, of shape (N, out_1, ... out_k)
        """
        out = x.dot(self.params["w"])+self.params["b"]
        mask = out<0
        out[mask] = 0
        self.cache = x, mask
        return out
    
    def backward(self, dx):
        """
            Computes the backward pass for an linear layer.

            Inputs:
            - dx: Upstream gradient data, of shape (N, out_1, ... out_k)
        """
        x, mask = self.cache
        dx = dx.copy()[mask]=0
        self.grads["w"] += x.T.dot(dx)
        self.grads["b"] += np.sum(dx, axis=0)
        dout = dx.dot(self.w.T)
        return dout


class Stack(Layer):
    """
       Stack layer class which is to stack given number of layers sequentially 
    """
    
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout