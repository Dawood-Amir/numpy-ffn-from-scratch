#ffn (w, b forward pass)
import numpy as np
from .activations import relu, softmax
import pickle

class FFN:
    '''# If i was doing Transformer-style FFN on Iris, it'd be :4 -> 16 -> 4 (d_model → 4 * d_model → d_model) but 
    #but here because its classification we just use 4->16->3 (3 classes)
    # hidden sizes are number of neurons in hidden layers
    # which we are expanding our input features to in order to learn more complex patterns in the data
    # and we compress it back to 4 orignal input feature size to then output classes for classification
    '''
    def __init__(self, input_size=4, hidden_size=16, hidden_size2=4, output_size=3):

        
        #w$ R^nin x nout  
        ## INITIALIZE WEIGHTS AND BIASES  
        #He Initialization srt2.0​​/inputsize
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)  # (4, 16)
        self.b1 = np.zeros((1,hidden_size))  # (1, 16), the output size should be( 150 or (batch_szie) , 16)

        self.w2 = np.random.randn(hidden_size, hidden_size2) * np.sqrt(2.0/hidden_size) # (16, 4)
        self.b2 = np.zeros((1, hidden_size2))  # (1, 16), the output size should be(150 or (batch_szie),4)

        self.w3 = np.random.randn(hidden_size2 , output_size) * np.sqrt(2.0/hidden_size2) # (4, 3) 
        self.b3 = np.zeros((1,output_size))  # (1, 3), the output size should be ( 150 or (batch_szie) , 3)

    def save(self, file_path:str ) -> None:
        """
        Saves the trained weights and biases to a file. 
        """
        model_state = {
            'w1': self.w1 , 'b1': self.b1 ,
            'w2': self.w2 , 'b1': self.b2 ,
            'w3': self.w1 , 'b1': self.b3 ,
            }
        
        with open(file_path ,'wb') as f :
             pickle.dump(model_state ,f)
        print(f" Model Saved succefully to {file_path}")


    def loadModel(self, file_path : str)-> None:# will use it later
        """
        Loads saved weights and biases into the current model instance.
        """
        with open(file_path , 'rb') as f :
             model_state =pickle.load(f)

        # Overwrite the random weights with our saved, highly trained weights
        self.w1 = model_state['w1']
        self.b1 = model_state['b1']
        self.w2 = model_state['w2']
        self.b2 = model_state['b2']
        self.w3 = model_state['w3']
        self.b3 = model_state['b3']
        print(f"Model weights successfully loaded ")
    

    # -------------------
    # FORWARD PASS
    # -------------------

    def forward(self,x):
        self.x= x
    
        # y = xW1 + b1
        self.z1= np.dot(x,self.w1)+ self.b1 # Linearity .. so here we explanded from orignal 4 feature
        #to 16 features per sample resulting in total shape of (x.shape[0], 16) = (150, 16) without batch 
        self.a1 = relu(self.z1)  # non-linearity (activation) .. so here we apply relu to get the activated output of hidden layer 1 which is also (batch_size, 16)
        # relu is good for hidden layers because it helps with vanishing gradients and is computationally efficient

        self.z2 = np.dot(self.a1, self.w2)+ self.b2 #output shape(150, 4) because we are compressing back to 4 features per sample
        self.a2 = relu(self.z2)  # (batch_size, 4) # could use GeLU for smoother activation
        #but relu is simpler and works fine for this example

        # z3 raw output layer raw scare 
        self.z3 = np.dot(self.a2 , self.w3) + self.b3 #output shuld be (150,3)
        # raw score to probability distribution 
        self.a3 = softmax(self.z3)
        return self.a3

    # -------------------
    # BACKPROP INTO HIDDEN
    # -------------------

    def backward(self, y_true ,lr=0.01):
        
        y_true = y_true.reshape(-1).astype(int)
        m = y_true.shape[0]

        # one-hot encoding converts it into matrix form:
        #y_onehot shape = (batch_size, num_classes) So: batch_size = 3 samples num_classes = 3 classes
        y_one_hot= np.zeros_like(self.a3) # (150, 3) or (batch_size, 3)

        """# BEFORE:
        # y_onehot =
        # [
        #   [0, 0, 0],
        #   [0, 0, 0],
        #   [0, 0, 0]...
        # ]"""
        
        y_one_hot[np.arange(m), y_true] = 1 # same label y_true but in vectored form now y_onehot is true ground labels 
        
        """# AFTER:
        # [
        #   [0, 0, 1],   ← class 2 is correct
        #   [1, 0, 0],   ← class 0 is correct
        #   [0, 1, 0]...    ← class 1 is correct
        # ]"""

        
        # -----------------------------------
        # 1. OUTPUT LAYER GRADIENTS (Layer 3)
        # -----------------------------------

        """# Why this is a nightmare in code (da3 = - (y_one_hot / self.a3) / m):
        # If you try to use this da3 directly, you will run into a massive problem: Dividing by Zero.
        # If your model predicts a probability of exactly 0 (self.a3 = 0) for a class, running 
        # y_one_hot / 0 will instantly give you NaN (Not a Number) or Infinity, crashing 
        # your training loop. 
        # so we do z3 (Raw Scores/Logits) Softmax ​a3 (Probabilities)-> Cross-Entropy​Loss together
        """
        
        #delta3 = a3 - y_onehot
        dz3 = (self.a3 - y_one_hot) / m # divide by m because we want the average gradient per sample across the batch
        dw3  = np.dot(self.a2.T , dz3)  
        db3 = np.sum(dz3 , axis=0, keepdims=True)  # (1, 3)  

        
        # -----------------------------------
        # 2. HIDDEN LAYER 2 GRADIENTS (Layer 2)
        # -----------------------------------
        
    
        # da2 = (a3T . delta3) # Gradient of loss w.r.t hidden layer 2 output 
        # delta2 =  da2 * Relu'(z2) # * here is Element-wise product Relu'(z2) = self.z2 > 0 because relu derivative is 1 for positive inputs and 0 for negative inputs
        da2 = np.dot(dz3 , self.w3.T ) 
        dz2 = da2 * (self.z2 > 0)  # ReLU derivative
        dw2 =  np.dot(self.a1.T , dz2) 
        db2 = np.sum(dz2, axis=0 , keepdims=True)  # (1, 4)
        
        # -----------------------------------
        # 2. HIDDEN LAYER 1 GRADIENTS (Layer 1)
        # -----------------------------------

        #da1    = (delta2 . dz2/da1 )
        #dz1    =  (da1 * da1/dz1)
        #dL/dw1 = x1T . dz1 

        da1 = np.dot(dz2 ,self.w2.T ) # gradient of loss w.r.t hidden layer 1 output
        dz1 = da1 * (self.z1 > 0)   # ReLU derivative for hidden layer 1
        dw1 = np.dot(self.x.T, dz1)  # dw1 = delta1 . x
        db1 = np.sum(dz1, axis=0, keepdims=True)
        '''#(self.z1 > 0) gives:
        # [
        #   [1, 0, 1, 1, ...],
        #   [0, 1, 1, 0, ...]
        # ]
        # meaning: only positive activations pass gradient 
        so only the neurons that were activated (positive) in the forward pass will receive gradients
        during backpropagation, which is a key feature of ReLU that helps with learning sparse 
        representations and mitigating the vanishing gradient problem.
        ''' 


        # -------------------
        # UPDATE WEIGHTS & BIASES Changing it from optimizer class 
        # -------------------
        self.dw3 = dw3
        self.db3 = db3
        self.dw2 = dw2
        self.db2 = db2
        self.dw1 = dw1
        self.db1 = db1


        # self.w3 -= lr * dw3
        # self.b3 -= lr * db3

        # self.w2 -= lr * dw2
        # self.b2 -= lr * db2

        # self.w1 -= lr * dw1
        # self.b1 -= lr * db1





    def zero_grad(self) -> None:
            """
            Clears the stored gradients from memory after the optimizer 
            has successfully updated the weights.
            """
            self.dw1 = None
            self.db1 = None
            self.dw2 = None
            self.db2 = None
            self.dw3 = None
            self.db3 = None