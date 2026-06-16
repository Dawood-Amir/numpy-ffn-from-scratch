import numpy as np 
from model.model import FFN


class SGD:# Vanilla sgd or with mini batch  W=W−α⋅dW
    def __init__(self , lr =0.01):
        self.lr = lr
        
    def step(self,model : FFN)-> None:
        #update weight and biases with mini batch here 
        model.w1 -= self.lr * model.dw1
        model.b1 -= self.lr * model.db1

        model.w2 -= self.lr * model.dw2
        model.b2 -= self.lr * model.db2
        
        model.w3 -= self.lr * model.dw3
        model.b3 -= self.lr * model.db3

        


class SGDMomentum: 
    '''
    To add velocity, we introduce a new hyperparameter called momentum (usually set to 0.9), denoted 
    as β (beta). Instead of updating the weights directly with the gradient, we update a velocity 
    vector (v) first:

    Update Velocity:
    v= β⋅v+lr⋅dw

    Update Weights:
    W= W - v
    
    '''
    def __init__(self , lr= 0.01 , momentum = 0.9):
        self.lr =  lr
        self.momentum = momentum

        # Dictionary to store the velocity (v) for each weight and bias
        self.v = {}
        
    def step(self, model : FFN) ->None :
        params = ['w1' , 'b1', 'w2' , 'b2', 'w3' , 'b3'] 

        for p in params: #placeholde p
            w = getattr(model,p)
            dw =getattr(model, 'd' + p) #joins 'd' + 'w1' to get 'dw1' etc

            # Initialize velocity matrix to zeros on the very first step
            if p not in self.v:
                self.v[p] = np.zeros_like(w)
                
            # 1. Calculate new velocity: 90% of old velocity + current step 
            self.v[p] = self.momentum * self.v[p] + self.lr * dw
            # 2. Apply velocity to the parameters (weights/biases)
            new_param = w - self.v[p]
            setattr(model, p, new_param)



class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Time step counter for bias correction
        
        # Dictionary to store first (m) and second (v) moments for every weight/bias
        self.m = {}
        self.v = {}

    def step(self, model):
        self.t += 1
        
        # List of parameter names and their current gradients matching the model attributes
        params = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']
        
        for p in params:
            # Get current weight matrix and its matching gradient
            w = getattr(model, p)
            dw = getattr(model, 'd' + p)
            
            # Initialize moments if this is the first step
            if p not in self.m:
                self.m[p] = np.zeros_like(w)
                self.v[p] = np.zeros_like(w)
                
            # Update biased first moment estimate (Momentum)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * dw
            # Update biased second raw moment estimate (Velocity)
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (dw ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            # Update the model's parameters
            new_w = w - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
            setattr(model, p, new_w)


class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = {}
        self.v = {}

    def step(self, model):
        self.t += 1
        params = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']
        
        for p in params:
            w = getattr(model, p)
            dw = getattr(model, 'd' + p)
            
            if p not in self.m:
                self.m[p] = np.zeros_like(w)
                self.v[p] = np.zeros_like(w)
                
            # 1. Standard Adam tracking steps
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * dw
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (dw ** 2)
            
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            # 2. Decoupled Weight Decay step (The 'W' magic happens right here!)
            # We penalize large weights directly before updating
            w = w - self.lr * self.weight_decay * w
            
            # 3. Apply the gradient update
            new_w = w - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
            setattr(model, p, new_w)



      