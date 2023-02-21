# Vectorflow
A simple light-weight neural network library developed using only numpy vectors.



```python
import numpy as np
from layers import *
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
    Conv2d(1, 16, 5, 2, 1),
    MaxPooling(4, 4, 2),
    Conv2d(16, 64, 3, 1, 1),
    MaxPooling(3, 3, 2),
    Conv2d(64, 64, 3, 1, 1),
    MaxPooling(4, 4, 2),
    Flatten(),
    Dropout(0.5),
    Linear(256, 256),
    ReLU(),
    Batchnorm(256),
    Dropout(0.5), 
    Linear(256, 10)
    
)
    
s = solver.Solver(model=model, opt_fn="adam", loss_fn="softmax", train_split = 0.9, batch_size=16, lr=0.001, lr_decay = 0.8)
s.train(X_train[:100,], y_train[:100,], epochs=10)
```

    Epoch:0, Training Loss:2.3570360814746296. Validation Loss:2.376750798273634
    Epoch:1, Training Loss:2.297280836753085. Validation Loss:2.380189907861677
    Epoch:2, Training Loss:2.178709546226372. Validation Loss:2.3769091121651758
    Epoch:3, Training Loss:2.1703757985761003. Validation Loss:2.3821536367708407
    Epoch:4, Training Loss:2.1066575163546473. Validation Loss:2.3884494233174935
    Epoch:5, Training Loss:2.1922979542544567. Validation Loss:2.3891626734066276
    Epoch:6, Training Loss:2.2853640714290617. Validation Loss:2.3901662656503353
    Epoch:7, Training Loss:2.182425535661223. Validation Loss:2.391083482067678
    Epoch:8, Training Loss:2.2269709853209894. Validation Loss:2.3902171303809547
    Epoch:9, Training Loss:2.209337999009011. Validation Loss:2.390505962063673



```python

```


```python

```


```python

```
