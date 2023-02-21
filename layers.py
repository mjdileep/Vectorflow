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
        
        self.params["w"] = np.random.randn(in_size, out_size)* np.sqrt(2/in_size) # He initialization
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
            

class ReLU(Layer):
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
        mask = x<0
        x[mask] = 0
        self.cache = mask
        return x
    
    def backward(self, dx):
        """
            Computes the backward pass for an linear layer.

            Inputs:
            - dx: Upstream gradient data, of shape (N, out_1, ... out_k)
        """
        mask = self.cache
        dx[mask]=0.0
        return dx

    
class Sigmoid(Layer):
    """
       Sigmoid layer class 
    """
    
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, d_1, ... d_k)

            Returns:
            - y: Output data, of shape (N, out_1, ... out_k)
        """
        x = 1/(1+np.exp(-x))
        self.cache = x
        return x
    
    def backward(self, dx):
        """
            Computes the backward pass for an linear layer.

            Inputs:
            - dx: Upstream gradient data, of shape (N, out_1, ... out_k)
        """
        x = self.cache
        dx = dx*x*(1-x)
        return dx

    
class Tanh(Layer):
    """
       Tanh layer class 
    """
    
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, d_1, ... d_k)

            Returns:
            - y: Output data, of shape (N, out_1, ... out_k)
        """
        e = np.exp(x)
        en = np.exp(-x)
        x = (e-en)/(e+en)
        self.cache = x
        return x
    
    def backward(self, dx):
        """
            Computes the backward pass for an linear layer.

            Inputs:
            - dx: Upstream gradient data, of shape (N, out_1, ... out_k)
        """
        x = self.cache
        dx = dx*(1-x**2)
        return dx
    
    
class Dropout(Layer):
    
    def __init__(self, keep=0.5):
        super().__init__()
        self.p = keep
    def forward(self, x, **kwargs):
        if kwargs["train"]:
            shape = x.shape
            x = x.reshape(shape[0], -1)
            mask = np.random.rand(x.shape[0], x.shape[1])>self.p
            x[mask] = 0
            x /=(self.p+0.0000001)
            self.cache = mask
            x = x.reshape(shape)
        return x
        
    def backward(self, dx):
        shape = dx.shape
        dx = dx.reshape(shape[0], -1)
        mask = self.cache
        dx[mask] = 0
        return dx.reshape(shape)
    
    
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
        dstd_sample_var = 0.5*((sample_var+self.eps)**(-0.5))
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
    """
       Batchnorm class 
    """
    def __init__(self, size, momentum=0.9, eps=1e-5):
        super().__init__(eps)
        
        self.params["gamma"] = np.random.randn(1, size) * np.sqrt(2/size)  # He initialization
        self.grads["gamma"] = np.zeros((1, size))
        self.param_configs["gamma"] = {}
        
        self.params["beta"] = np.random.randn(1, size) * np.sqrt(2/size) # He initialization
        self.grads["beta"] = np.zeros((1, size))
        self.param_configs["beta"] = {}
        
        self.momentum = momentum
        self.running_mean = 0.0
        self.running_var = 0.0
        
    def forward(self, x, **kwargs):
        r = int(x.shape[1]/self.params["gamma"].shape[1])
        gamma = np.repeat(self.params["gamma"], r)
        beta = np.repeat(self.params["beta"], r)
            
        if kwargs["train"]:
            norm, sample_mean, sample_var = self.norm(x)
            out = norm*gamma + beta
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var
        else:
            norm = (x - self.running_mean)/(np.sqrt(self.running_var)+self.eps)
            out = norm*gamma + beta
            
        return out
        
    def backward(self, dx):
        _, _, _, _, norm, _, _, _, _, _ = self.cache
        r = int(dx.shape[1]/self.params["gamma"].shape[1])
        gamma = np.repeat(self.params["gamma"], r)
        beta = np.repeat(self.params["beta"], r)
            
        dbeta = dx.sum(axis=0)
        dgamma = (norm*dx).sum(axis=0)
        dnorm = dx*gamma
        dnorm_dx_1, dnorm_dx_2, dnorm_dx_3 = self.dnorm(dnorm)

        dx = dnorm_dx_1 + dnorm_dx_2 + dnorm_dx_3
        
        self.grads["gamma"] = np.add.reduceat(dgamma, np.arange(0,dgamma.shape[0],r))
        self.grads["beta"] = np.add.reduceat(dbeta, np.arange(0,dbeta.shape[0],r))
        return dx
    
class Layernorm(Norm):
    """
       Layernorm class 
    """
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
        

class Conv2d(Layer):
    
    """
       Conv2d layer class 
    """
    
    def __init__(self, in_filters, out_filters, kernal_size=3, padding=1, stride=1):
        """
            Constructor of Conv2d Layer 
            Inputs:
            - in_filters: input filter size
            - out_filters: output filter size
            - kernal_size: Kernal size
            - padding: padding (zero padding)
            - stride: stride
        """
        super().__init__()
        
        self.params["w"] = np.random.randn(out_filters, in_filters, kernal_size, kernal_size)* np.sqrt(2/(in_filters*kernal_size**2))
        self.grads["w"] = np.zeros_like(self.params["w"])
        self.param_configs["w"] = {}
        
        self.params["b"] = np.zeros(out_filters)
        self.grads["b"] = np.zeros(out_filters)
        self.param_configs["b"] = {}
        
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride
        
        
        self.cache = None
        
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, C, H, W)

            Returns:
            - y: Output data, of shape (N, F, _H, _W)
        """
        #Shape W = (F, C, K, K)
        #out shape = (N, F, _H, _W)
        pad = self.padding
        stride = self.stride
        padded = np.pad(x,((0,0), (0,0), (pad,pad), (pad,pad)), constant_values=0)
         
        
        w = self.params["w"]
        b = self.params["b"]
        N, _, H, W = x.shape
        F, _, HH, WW = w.shape
        _H = int(1 + (H + 2 * pad - HH) / stride)
        _W = int(1 + (W + 2 * pad - WW) / stride)
        
        out = np.zeros((N, F, _H, _W))
        for n in range(N):
            for f in range(F):
                for i in range(0, H+2*pad - HH+1, stride):
                    for j in range(0, W+2*pad - WW+1, stride):
                        out[n,f,int(i/stride),int(j/stride)]= np.sum(padded[n,:,i:i+HH,j:j+WW]*w[f,:,:,:])+b[f]

        self.cache = padded
        return out
    
    def backward(self, dout):
        """
            Computes the backward pass for an conv2d layer.

            Inputs:
            - dx: Upstream gradient data, of shape (N, F, W, H)
        """
        padded = self.cache
        pad = self.padding
        stride = self.stride 
        N, F, W, H = dout.shape
        w = self.params["w"]
        b = self.params["b"]
        
        dw = np.zeros(w.shape)
        dx = np.zeros(padded.shape)
        k = w.shape[-1]
        for n in range(N):
            for f in range(F):
                for i in range(W):
                    for j in range(H):
                        dw[f,:,:,:] += dout[n,f,i,j] * padded[n,:, i*stride:i*stride+k, j*stride:j*stride+k]
                        dx[n,:,i*stride:i*stride+k, j*stride:j*stride+k] += dout[n,f,i,j] * w[f,:,:,:]

        dx = dx[:,:,pad:-pad, pad:-pad]
        db = np.sum(dout, axis=(0,2,3))
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx


class MaxPooling(Layer):
    
    """
       Pooling layer class 
    """
    
    def __init__(self, pool_height, pool_width, stride=1):
        """
            Constructor of Pooling Layer 
            Inputs:
            - pool_height: height of the pooling
            - pool_width: width of the pooling
            - padding: padding (zero padding)
            - stride: stride
        """
        super().__init__()
        
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        
        self.cache = None
        
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, C, H, W)

            Returns:
            - y: Output data, of shape (N, F, _H, _W)
        """
        out = None
        N, C, H, W = x.shape
        pool_height = self.pool_height
        pool_width = self.pool_width
        stride =self.stride

        assert (H - pool_height) % stride == 0
        assert (W - pool_width) % stride == 0
        _H = int(1 + (H - pool_height) / stride)
        _W = int(1 + (W - pool_width) / stride)
        out = np.zeros((N, C, _H, _W))
        for h in range(0,H-pool_height, stride):
            for w in range(0,W-pool_width, stride):
                out[:,:,int((h+1)/stride)-1, int((w+1)/stride)-1] = np.max(x[:,:,h:h+pool_height, w:w+pool_width], axis=(2,3))

        self.cache = x
        return out
    
    
    def backward(self, dout):
        """
            Computes the backward pass 

            Inputs:
            - dx: Upstream gradient data, of shape (N, F, W, H)
        """
        x = self.cache
        N, C, H, W = x.shape
        _, _, _H, _W = dout.shape
        pool_height = self.pool_height
        pool_width = self.pool_width
        stride =self.stride
        assert (H - pool_height) % stride == 0
        assert (W - pool_width) % stride == 0

        dx = np.zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for h in range(0,_H):
                    for w in range(0,_W):
                        chunk = x[n,c,h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]
                        ind = np.unravel_index(np.argmax(chunk, axis=None), chunk.shape)
                        dx[n,c,h*stride+ind[0], w*stride+ind[1]] += dout[n,c,h,w]
        return dx
    

class AvgPooling(Layer):
    
    """
       Pooling layer class 
    """
    
    def __init__(self, pool_height, pool_width, stride=1):
        """
            Constructor of Pooling Layer 
            Inputs:
            - pool_height: height of the pooling
            - pool_width: width of the pooling
            - padding: padding (zero padding)
            - stride: stride
        """
        super().__init__()
        
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        
        self.cache = None
        
    def forward(self, x, **kwargs):
        """
            Computes the forward pass.

            Inputs:
            - x: Input data, of shape (N, C, H, W)

            Returns:
            - y: Output data, of shape (N, F, _H, _W)
        """
        out = None
        N, C, H, W = x.shape
        pool_height = self.pool_height
        pool_width = self.pool_width
        stride =self.stride

        assert (H - pool_height) % stride == 0
        assert (W - pool_width) % stride == 0
        _H = int(1 + (H - pool_height) / stride)
        _W = int(1 + (W - pool_width) / stride)
        out = np.zeros((N, C, _H, _W))
        for h in range(0,H-pool_height, stride):
            for w in range(0,W-pool_width, stride):
                out[:,:,int((h+1)/stride)-1, int((w+1)/stride)-1] = np.mean(x[:,:,h:h+pool_height, w:w+pool_width], axis=(2,3))

        self.cache = x
        return out
    
    
    def backward(self, dout):
        """
            Computes the backward pass 

            Inputs:
            - dx: Upstream gradient data, of shape (N, F, W, H)
        """
        x = self.cache
        N, C, H, W = x.shape
        _, _, _H, _W = dout.shape
        pool_height = self.pool_height
        pool_width = self.pool_width
        stride =self.stride
        assert (H - pool_height) % stride == 0
        assert (W - pool_width) % stride == 0

        dx = np.zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for h in range(0,_H):
                    for w in range(0,_W):
                        dx[n,c,h*stride:h*stride+pool_height, w*stride:w*stride+pool_width] += dout[n,c,h,w]/(pool_height)
        return dx


class Batchnorm2d(Layer):
    def __init__(self, size, momentum=0.9, eps=1e-5):
        super().__init__()
        
        self._batchnorm = Batchnorm(size, momentum=momentum, eps=eps)
        
    def forward(self, x, **kwargs):
        shape = x.shape
        x = self._batchnorm.forward(x.reshape(shape[0],-1), **kwargs)
        return x.reshape(shape)
    
    def backward(self, dout):
        shape = dout.shape
        dout = self._batchnorm.backward(dout.reshape(shape[0],-1))
        return dout.reshape(shape)
        
    
    
class Flatten(Layer):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, **kwargs):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dx):
        return dx.reshape(self.cache)
    
    

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
