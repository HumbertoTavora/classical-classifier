import numpy as np

class Perceptron:
    def __init__(self, N, alpha = 0.1):
        
        self.w = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # step function
        if(x > 0):
            return 1
        else:
            return 0
    
    def fit(self, X, y, epochs = 10):
        # X is the training data
        # y is the target output class label
         
        X = np.c_[X, np.ones((X.shape[0]))] # trick to treat the bias as a trainable parameter within the weight matrix
        
        for epoch in np.arange(0,epochs):
            for(x, target) in zip(X,y):
                p = self.step(np.dot(x,self.W))
                
                if(p != target):    # If the prediction is wrong:
                    error = p - target      # Determine the error value
                    self.W += -self.alpha + error*x # Update te weight

    def predict(self, X, addBias = True):
        X = np.atleast_2d(X)  #ensure the matrix format

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))
