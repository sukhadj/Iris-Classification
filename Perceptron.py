import numpy as np

class perceptron:
 
    def __init__(self,eta=0.01,n_iters=10):
        self.eta=eta
        self.n_iters=n_iters
        
        
    def fit(self,X,y):
        """ X is matrix dimension of m*(n+1); y is matrix of m*1; theta is dimension of n+1*1  """
        self.theta=np.zeros(1+X.shape[1])
        self.error_list=[]
        for _ in range(self.n_iters):
            error=0;
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.theta[1:]+=update*xi
                self.theta[0]+=update
                error+=int(update!=0.0)
            self.error_list.append(error)
        return self
    def net_input(self,X):
        return(np.dot(X,self.theta[1:])+self.theta[0])


    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,-1)



    

print("Executing the program")
