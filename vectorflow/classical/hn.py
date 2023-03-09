"""
A Hopfield network (or Ising model of a neural network or Ising–Lenz–Little model) is a form of recurrent artificial neural network and a type of spin glass system popularised by John Hopfield in 1982[1] as described earlier by Little in 1974 based on Ernst Ising's work with Wil3helm Lenz on the Ising model. Hopfield networks serve as content-addressable ("associative") memory systems with binary threshold nodes, or with continuous variables. Hopfield networks also provide a model for understanding human memory.
https://en.wikipedia.org/wiki/Hopfield_network
#
"""
import numpy as np


class HopfieldNetwork:
    
    
    def __init__(self, input_size:int, size: int, max_no_change_count=10):
        self.connections = np.zeros((size,size))
        self.max_no_change_count = size*2
        self.max_itter = size**2
        self.n = size
        self.input_size = input_size
        self.units =  np.ones(size)
        self.temperature_start = 100
        
    def fit(self, data):
        assert data.shape[1] == self.input_size
        shuffle = np.arange(data.shape[0])
        np.random.shuffle(shuffle)
        data = np.copy(data)
        data = data[shuffle,:]
        for i in range(data.shape[0]):
            inputs = np.copy(data[i,:])
            self.units[:self.input_size] = inputs
            no_change_count = self.max_no_change_count
            itter = 0
            while no_change_count > 0 and itter < self.max_itter:
                r = np.random.randint(0, self.n)
                c = True
                while True:
                    val = sum([self.units[j]*self.connections[r, j] for j in range(0, self.n)])
                    p = 1/(1+np.exp(-val/max(1, self.temperature_start-itter)))
                    if int(np.random.rand()<p) == int(self.units[r]):
                        break
                    else:
                        k = np.random.randint(0,self.n)
                        self.connections[r, k] += (k!=r)*2*(2*self.units[r]-1)*(2*self.units[k]-1)
                        # 2*(2si-1)(2sj-1)
                        c = False
                if c:
                    no_change_count -=1
                itter +=1
                
                
            
    def retrieve(self, data):
        assert data.shape[0] <= self.n
        units = np.zeros(self.n)
        units[0:data.shape[0]] = data
        
        goodness_increase = 0
        for i in range(self.max_itter):
            r = np.random.randint(0, self.n)
            
            local_goodness = sum([self.connections[r, i]*units[i] for i in range(self.n)])
            if local_goodness <0 and units[r] !=0 :
                units[r] = 0
                goodness_increase -= local_goodness
                
            elif local_goodness >0 and units[r] !=1 :
                units[r] = 1
                goodness_increase += local_goodness
        
        return units[:self.input_size]
