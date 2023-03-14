import numpy as np


class RBM:
    """

    """
    def __init__(self, v, h):
        self.v = v
        self.h = h
        self.ww = np.random.randn(v, h)
        self.bb_h = np.zeros(h)
        self.bb_v = np.zeros(v)
        
        self.ss_h = np.zeros(h)
        self.ss_v = np.zeros(v)
        
    
    def __e(self, i, is_visible=True):
        """ Calculates energy per unit"""
        if is_visible:
            return np.dot(self.ww[i,:], self.ss_h) + self.bb_v[i]
        else:
            return np.dot(self.ww[:,i], self.ss_v) + self.bb_h[i]
        
    def reconstruct(self, start=0):
        for i in range(start, self.v):
            energy = self.__e(i, True)
            if energy > 0:
                self.ss_v[i] = 1
            elif energy < 0:
                self.ss_v[i] = 0
    
    def update_hidden(self):
        for i in range(self.h):
            energy = self.__e(i, False)
            if energy > 0:
                self.ss_h[i] = 1
            elif energy < 0:
                self.ss_h[i] = 0
    
    def fit(self, data, epochs=10):
        d, v = data.shape
        assert v == self.v, "Visible vector size doesn't match:{}!={}".format(v, self.v)
        h = self.h
        for epoch in range(epochs):
            
            start_hidden_states = np.empty((d, h))
            start_visible_states = np.empty((d, v))
            
            end_hidden_states = np.empty((d, h))
            end_visible_states = np.empty((d, v))
            
            for point in range(d):
                self.ss_v = np.copy(data[point,:])
                self.update_hidden()
                
                # Collect states
                start_hidden_states[point,:] = np.copy(self.ss_h)
                start_visible_states[point,:] = np.copy(self.ss_v)
                
                f = int(np.log(3+epoch**5))
                
                for i in range(f):
                    self.reconstruct()
                    self.update_hidden()
                
                # Collect states
                end_hidden_states[point,:] = np.copy(self.ss_h)
                end_visible_states[point,:] = np.copy(self.ss_v)
            
            delta_ww = (np.dot(start_visible_states.T, start_hidden_states) - np.dot(end_visible_states.T, end_hidden_states))/d
            self.ww += delta_ww
            
            delta_bb_h = np.mean(start_hidden_states, axis=0) - np.mean(end_hidden_states, axis=0)
            delta_bb_v = np.mean(start_visible_states, axis=0) - np.mean(end_visible_states, axis=0)
            self.bb_h += delta_bb_h
            self.bb_v += delta_bb_v
                

    def retrieve(self, vector):
        v = vector.shape[0]
        assert v <= self.v, "Input vector size is not compatible match:{}>{}".format(v, self.v)
        self.ss_v = np.full(self.v, False) 
        self.ss_v[0:v] = np.copy(vector)
        self.update_hidden()
        self.reconstruct(start=0)
        return np.copy(self.ss_v)
