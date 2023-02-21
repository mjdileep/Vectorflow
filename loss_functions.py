import numpy as np


def mse(pred, y, mode="train"):
    """ 
        Mean Squard Error loss fucntion 
        Inputs:
        - pred: predicted values
        - y: expected outputs
    """
    assert pred.shape == y.shape, "Dimensions are not matching!"
    loss = np.sum((pred-y)**2/pred.shape[0])
    dout = None
    if mode=="train":
        dout = 2*(pred-y)/pred.shape[0]
    return loss, dout


def softmax(pred, y, mode="train"):
    """
        Softmax loss fucntion 
        Inputs:
        - pred: predicted values (N, D)
        - y: expected output indices (N,)
    """
    N, _ = pred.shape
    e = np.exp(pred)
    s = np.sum(e, axis=1, keepdims=True)
    norm = e/s
    scores = norm[np.arange(N),y]
    loss = np.sum(-np.log(scores))/N
    dout = None
    if mode=="train":
        dout = np.zeros(pred.shape)
        dout[np.arange(N),y]=1
        dout = (e/s - dout)/N
    return loss, dout


