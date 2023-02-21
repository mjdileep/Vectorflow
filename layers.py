import numpy as np


class Layer:
    """
        The base class of the Layer 
    """
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.param_configs = {}
    
    
    def step(self, fn, lr):
        for key in self.params:
            self.param_configs[key]["learning_rate"] = lr
            self.params[key], self.param_configs[key] = fn(self.params[key], self.grads[key], self.param_configs[key])
            
        
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def grad_zero(self):
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
        super().__init__()
        
        self.params["w"] = np.random.randn(in_size, out_size)* np.sqrt(2/(in_size**2+out_size**2)) # He initialization
        self.grads["w"] = np.zeros((in_size, out_size))
        self.param_configs["w"] = {}
        
        self.params["b"] = np.zeros(out_size)
        self.grads["b"] = np.zeros(out_size)
        self.param_configs["b"] = {}
        
        
        self.cache = None
        
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
        dout = dx.dot(self.params["w"].T)
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
        dx = np.copy(dx)
        dx[mask]=0.0
        self.grads["w"] += x.T.dot(dx)
        self.grads["b"] += np.sum(dx, axis=0)
        dout = dx.dot(self.params["w"].T)
        return dout

class Dropout(Layer):
    
    def __init__(self, keep=0.5):
        super().__init__()
        self.p = keep
    def forward(self, x, **kwargs):
        if kwargs["train"]:
            mask = np.random.rand(x.shape[0], x.shape[1])>self.p
            x[mask] = 0
            x /=(self.p+0.0000001)
            self.cache = mask
        return x
        
    def backward(self, dx):
        mask = self.cache
        dx[mask] = 0
        return dx
    
    
class Norm(Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def norm(self, x):
        n = x.shape[0]
        x_sum = x.sum(axis=0)
        sample_mean = x_sum/n
        x_c = x- sample_mean
        x_d = x_c**2
        x_e = x_d.sum(axis=0)
        sample_var = x_e/n
        std = sample_var**0.5 +self.eps
        x_a = x-sample_mean
        norm = x_a/std
        self.cache = x, std, sample_mean, sample_var, norm, x_sum, x_c, x_d, x_e, x_a
        return norm, sample_mean, sample_var
    
    def dnorm(self, dnorm):
        n = dnorm.shape[0]
        x, std, sample_mean, sample_var, norm, x_sum, x_c, x_d, x_e, x_a = self.cache
        dnorm_dx_a = dnorm/std
        dnorm_dx_1 = dnorm_dx_a

        dnorm_std = -dnorm*x_a/(std**2)
        dstd_sample_var = 0.5*(sample_var**(-0.5))
        dnorm_sample_var = np.sum(dnorm_std*dstd_sample_var, axis=0)
        dsample_var_x_e = 1/n
        dnorm_x_e = dnorm_sample_var*dsample_var_x_e
        dx_e_x_d = 1.0
        dnorm_x_d = dx_e_x_d*dnorm_x_e
        dx_d_x_c = 2.0*x_c
        dnorm_x_c = dnorm_x_d * dx_d_x_c
        dx_c_x = 1.0
        dnorm_dx_2 = dnorm_x_c* dx_c_x

        dx_c_sample_mean = -1.0
        dnorm_sample_mean =  np.sum(dnorm_x_c * dx_c_sample_mean-dnorm_dx_a,axis=0)
        dsample_mean_x_sum = 1.0/n
        dnorm_x_sum = dnorm_sample_mean * dsample_mean_x_sum
        dx_sum_x = 1.0
        dnorm_dx_3 = dnorm_x_sum*dx_sum_x
        
        return dnorm_dx_1, dnorm_dx_2, dnorm_dx_3

    
class Batchnorm(Norm):
    
    def __init__(self, size, momentum=0.9, eps=1e-5):
        super().__init__(eps)
        
        self.params["gamma"] = np.random.randn(1, size)* np.sqrt(2/(size**2)) # He initialization
        self.grads["gamma"] = np.zeros((1, size))
        self.param_configs["gamma"] = {}
        
        self.params["beta"] = np.random.randn(1, size)* np.sqrt(2/(size**2)) # He initialization
        self.grads["beta"] = np.zeros((1, size))
        self.param_configs["beta"] = {}
        
        self.momentum = momentum
        self.running_mean = np.zeros((1, size))
        self.running_var = np.zeros((1, size))
        
    def forward(self, x, **kwargs):
        if kwargs["train"]:
            norm, sample_mean, sample_var = self.norm(x)
            out = norm*self.params["gamma"] + self.params["beta"]
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var
        else:
            norm = (x - self.running_mean)/(np.sqrt(self.running_var)+self.eps)
            out = norm*self.params["gamma"] + self.params["beta"]
            
        return out
        
    def backward(self, dx):
        _, _, _, _, norm, _, _, _, _, _ = self.cache
        dbeta = dx.sum(axis=0)
        dgamma = (norm*dx).sum(axis=0)
        dnorm = dx*self.params["gamma"]
        dnorm_dx_1, dnorm_dx_2, dnorm_dx_3 = self.dnorm(dnorm)

        dx = dnorm_dx_1 + dnorm_dx_2 + dnorm_dx_3
    
        self.grads["gamma"] = dgamma
        self.grads["beta"] = dbeta
        return dx
    
class Layernorm(Norm):
    
    def __init__(self, size, momentum=0.9, eps=1e-5):
        super().__init__(eps)
        self.params["gamma"] = np.random.randn(1, size)* np.sqrt(2/(size**2)) # He initialization
        self.grads["gamma"] = np.zeros((1, size))
        self.param_configs["gamma"] = {}
        
        self.params["beta"] = np.random.randn(1, size)* np.sqrt(2/(size**2)) # He initialization
        self.grads["beta"] = np.zeros((1, size))
        self.param_configs["beta"] = {}
        
    def forward(self, x, **kwargs):
        x_T = x.T
        norm, sample_mean, sample_var = self.norm(x_T)
        out = norm.T*self.params["gamma"] + self.params["beta"]
        return out
    
    def backward(self, dx):
        _, _, _, _, norm, _, _, _, _, _ = self.cache
        self.grads["beta"] = dx.sum(axis=0)
        self.grads["gamma"] = (norm.T*dx).sum(axis=0)
        
        dnorm_dx_1, dnorm_dx_2, dnorm_dx_3 = self.dnorm(dx.T)
        
        return dnorm_dx_1.T + dnorm_dx_2.T + dnorm_dx_3.reshape(dx.shape[0], -1)
        


class Stack:
    """
       Stack layer class which is to stack given number of layers sequentially 
       
    """
    
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def grad_zero(self):
        """
            Set accumilated gradients to zero in the each layer 
        """
        for layer in self.layers:
            layer.grad_zero()
            
    def step(self, opt_fn, lr):
        for layer in self.layers:
            layer.step(opt_fn, lr)
    
    def predict(self, x):
        return self(x, train=False)
