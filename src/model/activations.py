import numpy as np

def relu(x):
    # ReLU activation functionif x < 0-> 0 x>=0 -> x kills negative values, keeps positive values unchanged
    return np.maximum(0, x)

def softmax(x):
    # Softmax activation function  sfotmax(x_i) = exp(x_i) / sum(exp(x_j))^exp()
    # convert raw scores (logits) into probabilities that sum to 1 across classes
   
    exp_x = np.exp(x - np.max(x,axis=1, keepdims=True))  # stability improvement
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # normalized probabilities 