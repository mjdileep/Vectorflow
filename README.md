# Vectorflow
A simple light-weight neural network library developed using only numpy vectors.



```python
import numpy as np
from vectorflow.layers import *
import solver
import scipy.io
from load_data import download
%load_ext autoreload
%autoreload 2
```


```python
# Load the MNIST dataset
download("./mnist-original.mat", "https://raw.githubusercontent.com/amplab/datascience-sp14/master/lab7/mldata/mnist-original.mat", False)
mat = scipy.io.loadmat('mnist-original.mat')
X = mat["data"].T
y_train = mat["label"].T.reshape(X.shape[0],).astype(int)
X_train = X.reshape(X.shape[0], 1, 28, 28)
randomize = np.arange(X.shape[0])
np.random.shuffle(randomize)
X_train = X_train[randomize]
y_train = y_train[randomize]
```

    ./mnist-original.mat already exists, skipping ...



```python
model = Stack(
    Conv2d(1, 16, 5, 2, 2),
    MaxPooling(4, 4, 1),
    Conv2d(16, 64, 3, 1, 2),
    MaxPooling(3, 3, 1),
    Flatten(),
    Dropout(0.5),
    Linear(1024, 1024),
    ReLU(),
    Batchnorm(1024),
    Dropout(0.5), 
    Linear(1024, 10)
    
)
    
s = solver.Solver(model=model, opt_fn="adam", loss_fn="softmax", train_split = 0.9, batch_size=256, lr=0.005, lr_decay = 0.8)
s.train(X_train[:10000,], y_train[:10000,], epochs=100)
```

    Epoch:0, Training Loss:0.2348860815666288. Validation Loss:0.32436609858848237
    Epoch:1, Training Loss:0.04375774714856664. Validation Loss:0.133075708786543
    Epoch:2, Training Loss:0.03635821863153767. Validation Loss:0.1428711216323174
    Epoch:3, Training Loss:0.05512305372606856. Validation Loss:0.11321439361176244
    Epoch:4, Training Loss:0.011365990897911163. Validation Loss:0.10776204321544039
    Epoch:5, Training Loss:0.029147142660130593. Validation Loss:0.08679076807792895
    Epoch:6, Training Loss:0.019303030819439293. Validation Loss:0.08161995388999545
    Epoch:7, Training Loss:0.03240544256062713. Validation Loss:0.08230960713510978
    Epoch:8, Training Loss:0.031936587603446. Validation Loss:0.0929233882230264
    Epoch:9, Training Loss:0.012377674451923778. Validation Loss:0.08273274130442544
    Epoch:10, Training Loss:0.009741044750975118. Validation Loss:0.07759969155440145
    Epoch:11, Training Loss:0.010705301830688078. Validation Loss:0.07934087237549542
    Epoch:12, Training Loss:0.05019445366062634. Validation Loss:0.07826549680301553
    Epoch:13, Training Loss:0.00759399152670637. Validation Loss:0.07764810472422692



```python

```


```python

```


```python

```
