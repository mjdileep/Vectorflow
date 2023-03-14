import numpy as np


class NaiveBM:
    """
    This is based on the most inefficient way to train the model as Geoffrey Hinton suggested in 1983. 
    https://en.wikipedia.org/wiki/Boltzmann_machine
    """
    def __init__(self, n, v, training_method=0, lr=1.0):
        self.lr = lr
        self._lambda = 0.9
        self.training_method = training_method # 0 for stochastic and 1 for meanfield
        self.v = v
        self.n = n
        self.bb = np.zeros(n)
        self.ss = np.random.rand(n) <0.5
        ww = np.random.randn(n,n)
        self.ww = (ww + ww.T)/2
        self.ww[np.arange(n),np.arange(n)]=0
        self.settle_itter = n**2
        self.partial_settle_itter = 3
        
    
    def __e(self, i): 
        """ Calculates energy per unit"""
        return np.dot(self.ww[i,:], self.ss) + self.bb[i]
    
    def __p(self, i, T=1):
        """ Calculates the probabilitt that a unit is on"""
        return 1/(1+np.exp(-self.__e(i)/T))
    
    
    def meanfield_settle(self, start=0, T=1):
        delta_p =1
        for i in range(self.settle_itter):
            r = np.random.randint(start,self.n)
            p = self.__p(r, T)
            delta_p = 0.5*delta_p + 0.5*abs(p-self.ss[r])
            self.ss[r] = self._lambda*p + (1-self._lambda)*self.ss[r]
            if abs(delta_p) < 0.01:
                break
    
    def partial_settle(self, start=0, T=1):
        for i in range(self.partial_settle_itter):
            r = np.random.randint(start,self.n)
            self.ss[r] = np.random.rand() < self.__p(r, T)
    
    def settle(self, start=0, T=4):
        tot_e_l = 0
        for i in range(self.settle_itter):
            r = np.random.randint(start,self.n)
            self.ss[r] = np.random.rand() < self.__p(r, T)
                
            if i%100 ==0:
                T = max(1, T-1)
                tot_e = sum([self.__e(i) for i in range(self.n)])
                #print("Energy:{}, T:{}, I:{}".format(tot_e, T, i))
                
                if abs(tot_e_l-tot_e) <0.0001:
                    return
                else:
                    tot_e_l = tot_e
        
    
    def fit(self, data, epochs=10):
        d, v = data.shape
        assert v == self.v, "Visible vector size doesn't match:{}!={}".format(v, self.v)
        n = self.n
        
        if self.training_method == 0:
            particles = np.random.rand(d, n) < 0.5 # Generate random vectors
        else:
            particles = np.full((d,n),0.5) # initilise with 0.5 
            
        particles[np.arange(d),0:v] = data # Clamp visible vectors
        
        fp = max(20,n*2)
        fantacy_particles =  np.random.rand(fp, n) < 0.5 # Generate random vectors
        
        for epoch in range(epochs):
            T = np.log(2+epochs-epoch) #Temperature
            # With visible vestors clamped 
            probs_v = np.empty((d, n))
            probs_v_w = np.empty((d, n, n))
            for each in range(d):
                if self.training_method == 0: #stochasitic method
                    self.ss = particles[each,:]
                    self.partial_settle(v, T=T) # Settle to thermal equilibrium using stochasitic method
                else: #meanfield method
                    self.ss = np.copy(particles[each,:])
                    self.meanfield_settle(v, T=T) # Settle to thermal equilibrium using meanfield method

                # Collect states
                probs_v[each,:] = np.copy(self.ss)
            
            # Get the probabilities of the combinations
            for i in range(d):
                probs_v_w[i,:,:] = np.outer(probs_v[i,:],probs_v[i,:]) 
            probs_v_w = np.mean(probs_v_w, axis=0)
            probs_v_b = np.mean(probs_v, axis=0)


            # With visible vestors not clamped 
            probs_nv = np.empty((fp, n))
            probs_nv_w = np.empty((fp, n, n))
            for each in range(fp):
                self.ss = fantacy_particles[each,:]
                self.partial_settle(T=T) # Settle to thermal equilibrium
                fantacy_particles[each,:] = self.ss # Remember the particle state

                # Collect states
                probs_nv[each, :] = np.copy(self.ss)

            # Get the probabilities of the combinations
            for i in range(fp):
                probs_nv_w[i,:,:] = np.outer(probs_nv[i,:],probs_nv[i,:]) 
            probs_nv_w = np.mean(probs_nv_w, axis=0)
            probs_nv_b = np.mean(probs_nv, axis=0)

            delta_ww = probs_v_w - probs_nv_w
            delta_bb = probs_v_b - probs_nv_b

            # Update weights
            self.ww += delta_ww*self.lr
            self.ww[np.arange(n),np.arange(n)]=0

            #Update bias
            self.bb += delta_bb*self.lr
            
            #print(delta_ww)

    def retrieve(self, vector, temperature=4):
        v = vector.shape[0]
        assert v <= self.v, "Input vector size is not compatible match:{}>{}".format(v, self.v)
        self.ss = np.random.rand(self.n) <0.5
        self.ss[0:v] = vector
        self.settle(v,temperature)
        return np.copy(self.ss[:self.v])