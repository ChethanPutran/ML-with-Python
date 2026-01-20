import numpy as np

class RNN:
    def __init__(self,input_size,output_size,memory_size=10):
        self.input_size = input_size
        self.output_size = output_size
        self.Wh = np.random.rand((memory_size,memory_size))
        self.Wx = np.random.rand((memory_size, input_size))
        self.Wy = np.random.rand((output_size,memory_size))
        self.h = np.random.rand((memory_size,1))
        self.f = np.tanh
        
    def forward(self,X):
        self.h = self.f(self.Wh @ self.h + self.Wx @ X)
        y = self.f(self.Wy @ self.h)
        return y

    def backward(self,X):
        pass
