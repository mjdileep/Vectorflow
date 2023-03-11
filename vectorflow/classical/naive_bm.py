import numpy as np


class NaiveBM:
    """
    This is based on the most inefficient way to train the model as Geoffrey Hinton suggested in 1983. 
    https://en.wikipedia.org/wiki/Boltzmann_machine
    """
    def __init__(self, n, v, temperature=4):
        self.T=temperature
        self.lr = 1.0
        self.v = v
        self.n = n
        self.bb = np.zeros(n)
        self.ss = np.random.rand(n) <0.5
        ww = np.random.randn(n,n)
        self.ww = (ww + ww.T)/2
        self.ww[np.arange(n),np.arange(n)]=0
        self.settle_itter = n**2
        
    
    def __e(self, i): 
        """ Calculates energy per unit"""
        return np.dot(self.ww[i,:], self.ss) + self.bb[i]
    
    def __p(self, i, T=1):
        """ Calculates the probabilitt that a unit is on"""
        return 1/(1+np.exp(-self.__e(i)/T))
    
    def settle(self, start=0):
        tot_e_l = 0
        T = self.T
        for i in range(self.settle_itter):
            r = np.random.randint(start,self.n)
            if self.__e(r) > 0 and np.random.rand()< self.__p(r, T):
                self.ss[r] = 1
            if i%100 ==0:
                T = max(1, T-1)
                tot_e = sum([self.__e(i) for i in range(self.n)])
                #print("Energy:{}, T:{}, I:{}".format(tot_e, T, i))
                
                if abs(tot_e_l-tot_e) <0.0001:
                    return
                else:
                    tot_e_l = tot_e
        
    
    def fit(self, data):
        d, v = data.shape
        assert v == self.v, "Visible vector size doesn't match:{}!={}".format(v, self.v)
        n = self.n
        
        # With visible vestors clamped 
        probs_v = np.empty((d, n))
        probs_v_w = np.empty((d, n, n))
        for each in range(d):
            self.ss = np.random.rand(n) <0.5
            self.ss[0:v] = data[each,:]
            self.settle(v) # Settle to thermal equilibrium
            
            # Collect probabilities
            for i in range(n):
                probs_v[each,i] = self.__p(i)
                
        # Get the probabilities of the combinations
        for i in range(d):
            probs_v_w[i,:,:] = np.outer(probs_v[i,:],probs_v[i,:]) 
        probs_v_w = np.mean(probs_v_w, axis=0)
        probs_v_b = np.mean(probs_v, axis=0)
        
        
        # With visible vestors not clamped 
        probs_nv = np.empty((n, n))
        probs_nv_w = np.empty((n, n, n))
        for each in range(n):
            self.ss = np.random.rand(n) <0.5
            self.settle() # Settle to thermal equilibrium
            
            # Collect probabilities
            for i in range(n):
                probs_nv[each, i] = self.__p(i)
        
        # Get the probabilities of the combinations
        for i in range(n):
            probs_nv_w[i,:,:] = np.outer(probs_nv[i,:],probs_nv[i,:]) 
        probs_nv_w = np.mean(probs_nv_w, axis=0)
        probs_nv_b = np.mean(probs_nv, axis=0)
        
        delta_ww = probs_v_w-probs_nv_w
        delta_bb = probs_v_b - probs_nv_b
        
        # Update weights
        self.ww += delta_ww
        self.ww[np.arange(n),np.arange(n)]=0
        
        #Update bias
        self.bb += delta_bb
    
    def retrieve(self, vector):
        v = vector.shape[0]
        assert v <= self.v, "Input vector size is not compatible match:{}>{}".format(v, self.v)
        self.ss = np.random.rand(self.n) <0.5
        self.ss[0:v] = vector
        self.settle(v)
        return np.copy(self.ss[:self.v])