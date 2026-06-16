
import numpy as np
import matplotlib.pyplot as plt
from model.model import FFN
from data_loader import get_data
import pickle
from model.optimizers import SGD, SGDMomentum, Adam, AdamW 


def accuracy(pred, y):
    return np.mean(np.argmax(pred, axis=1) == y.reshape(-1))

X_train, X_test, y_train, y_test = get_data()

# Configuration
epochs = 150
batch_size = 32

# Define the experiments we want to run
experiments = {
    "Vanilla SGD (lr=0.005)": SGD(lr=0.005),
    "SGD With Momentum (lr=0.005)": SGDMomentum(lr=0.005),
    "Adam (lr=0.005)": Adam(lr=0.005),
    "AdamW (lr=0.005)": AdamW(lr=0.005, weight_decay=0.01)
}

results = {}
best_val_acc = -1.0  # Track the highest accuracy seen so far
best_optimizer_name = ""
# Run each optimizer experiment
for name, optimizer in experiments.items():
    print(f"\nTraining with {name}...")
    model = FFN()  # Fresh model with new random weights for every test
    

    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Batch loop
        for i in range(0, X_train.shape[0], batch_size):
            X_b = X_shuffled[i : i + batch_size]
            y_b = y_shuffled[i : i + batch_size]
            
            # 1. Forward Pass: Calculate predictions
            model.forward(X_b)

            # Backward Pass: Gradients are calculated and saved inside model
            model.backward(y_b)  

            # 3. Optimizer Step: Read public gradients and update weights/biases
            optimizer.step(model) 

            # 4. Clear the memory! Detach gradients until the next batch runs
            model.zero_grad()  

            

        # --- VALIDATION EVALUATION ---
        # 1. Evaluate on Training Data
        train_preds = model.forward(X_train)
        train_acc = accuracy(train_preds, y_train)
        train_history.append(train_acc)

        # 2. Evaluate on Test (Validation) Data without backward pass!
        validation_pred = model.forward(X_test)
        validation_acc = accuracy(validation_pred , y_test)
        val_history.append(validation_acc)
        
        # MOVE THE CHECK HERE: Check every single epoch for a new peak!
        # Track which model is the best right here in memory
        if validation_acc >= best_val_acc:
            best_val_acc = validation_acc
            best_optimizer_name = name
            
            # Take a silent snapshot of the current best arrays
            best_w1, best_b1 = model.w1.copy(), model.b1.copy()
            best_w2, best_b2 = model.w2.copy(), model.b2.copy()
            best_w3, best_b3 = model.w3.copy(), model.b3.copy()

    
        
    results[name] = {"train" : train_history , "val" : val_history  }
    test_preds = model.forward(X_test)
    print(f"{name} Final Train Acc: {train_history[-1]:.3f} | Final Validation Acc: {val_history[-1]:.3f}")
print(f"\n The best optimizer was {best_optimizer_name} with a Val Acc of {best_val_acc:.3f}!")

best_model_state = {
    'w1': best_w1, 'b1': best_b1,
    'w2': best_w2, 'b2': best_b2,
    'w3': best_w3, 'b3': best_b3
}
with open("best_iris_model.pkl", "wb") as f:
    pickle.dump(best_model_state, f)
print(" Best model saved cleanly to disk!")

# -------------------------------------------------------------
# PLOT THE TRAINING VS VALIDATION ANALYSIS 
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# Plot 1: Training Convergence
for name, history in results.items():
    ax1.plot(history["train"], linewidth=2, label=name)
ax1.set_title("Training Set Accuracy Comparison", fontsize=12, fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

# Plot 2: Validation Generalization
for name, history in results.items():
    ax2.plot(history["val"], linewidth=2, label=name, linestyle="--")
ax2.set_title("Validation Set Accuracy (Generalization)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Optimizer Benchmarking suite (NumPy FFN From Scratch)", fontsize=14, y=0.98)
plt.savefig("optimizer_validation_comparison.png", dpi=300)
plt.show()