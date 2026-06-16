  # training loop, use main.py file for different optimizers

import numpy as np
from model.model import FFN
from data_loader import get_data

def accuracy(pred, y):
    pred_class = np.argmax(pred, axis=1)
    y = y.reshape(-1).astype(int)
    return np.mean(pred_class == y)

X_train, X_test, y_train, y_test = get_data()


model = FFN()

epochs = 200
lr = 0.01
batch_size = 32

num_samples = X_train.shape[0] #150

for epoch  in range(epochs):

    # -------------------------------------------------------------
    #  SHUFFLE DATA AT THE START OF EVERY EPOCH
    # -------------------------------------------------------------
    # This mixes up the data rows so the model doesn't learn sequence patterns

    shuffled_indices =  np.random.permutation(num_samples) # shuffle indices
    X_train_shuffled = X_train[shuffled_indices] # shuffle data
    y_train_shuffled = y_train[shuffled_indices] # shuffle labels
    # -------------------------------------------------------------
    # STEP B: LOOP OVER MINI-BATCHES
    # -------------------------------------------------------------


    for i in range(0 , num_samples ,batch_size):
        x_batch = X_train_shuffled[ i : i+batch_size]
        y_batch = y_train_shuffled[i : i + batch_size]

        # forward pass with mini batch
        model.forward(X_train)

        # # Backward pass & weight update on the mini-batch
        model.backward(y_train, lr)
    

    if epoch % 20 == 0:
        # We pass the entire X_train here just to print a true snapshot of accuracy
        full_train_preds = model.forward(X_train)
        acc = accuracy(full_train_preds, y_train)
        print(f"Epoch {epoch}, Train Accuracy: {acc:.3f}")



# test
test_preds = model.forward(X_test)
print("-" * 30)
print("Final Test Accuracy:", accuracy(test_preds, y_test))