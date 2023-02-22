import optim
import .loss_functions
import numpy as np

class Solver:
    def __init__(self, model, opt_fn="adam", loss_fn="softmax", train_split = 0.9, batch_size=64, lr=1e-3, lr_decay = 0.6):
        self.model = model
        self.opt_fn = getattr(optim, opt_fn)
        self.loss_fn = getattr(loss_functions, loss_fn)
        self.train_split = train_split
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        
    def train(self, X_train, y_train, epochs=10):
        
        N = y_train.shape[0]
        n = int(N*self.train_split)
        choices = np.random.choice(N, n,  replace=False)
        _choices = np.setdiff1d(np.arange(N), choices)
        
        X = np.take(X_train, choices, axis=0)
        y = np.take(y_train, choices, axis=0)
        
        X_val = np.take(X_train, _choices, axis=0)
        y_val = np.take(y_train, _choices, axis=0)
        
        for epoch in range(epochs):
            for i in range(0, n, self.batch_size):
                x_mini = X.take(indices=range(i, min(n,i+self.batch_size)), axis=0)
                y_mini = y.take(indices=range(i, min(n,i+self.batch_size)), axis=0)
                
                pred = self.model(x_mini, train=True)
                loss, dx = self.loss_fn(pred, y_mini, "train")
                self.model.backward(dx)
                self.model.step(self.opt_fn, self.lr)
                self.model.grad_zero()
            self.lr *= self.lr_decay
            pred = self.model.predict(X_val)
            valid_loss, _ = self.loss_fn(pred, y_val, "test")
            print("Epoch:{}, Training Loss:{}. Validation Loss:{}".format(epoch, loss, valid_loss))