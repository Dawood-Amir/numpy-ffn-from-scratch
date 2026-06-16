# Custom Feedforward Neural Network (FFN) From Scratch in NumPy

A production-ready, object-oriented implementation of a multi-layer Feedforward Neural Network (FFN) built entirely from scratch using Python and NumPy. This project avoids high-level deep learning frameworks such as PyTorch and TensorFlow to demonstrate the underlying mathematics of neural networks, including forward propagation, manual backpropagation, gradient-based optimization, weight initialization, checkpointing, and inference deployment.

---

# 🚀 Key Features

### Pure NumPy Implementation

Every component of the neural network is implemented directly with NumPy, including matrix multiplications, activation functions, gradient calculations, and parameter updates.

### Modular Optimizer Framework

Supports multiple optimization algorithms for performance comparison and convergence analysis:

* Stochastic Gradient Descent (SGD)
* SGD with Momentum
* Adam
* AdamW

### He (Kaiming) Weight Initialization

Weights are initialized using variance scaling based on layer dimensions, helping maintain signal propagation and preventing vanishing or exploding gradients.

### Model Checkpointing

Automatically tracks the highest validation accuracy during training and saves the best-performing parameter set as a serialized `.pkl` checkpoint.

### Production-Ready Inference

Provides a standalone inference pipeline capable of loading trained weights and performing classification without requiring a training graph or gradient calculations.

---

# 📊 Optimizer Benchmarking Performance

The network was trained on the standardized Iris dataset for **150 epochs** using a **mini-batch size of 32**. Both training and validation accuracy are tracked throughout training to evaluate convergence speed and generalization performance.

## Convergence Analysis

![Optimizer Comparison](optimizer_validation_comparison.png)

### Experimental Highlights

#### Adam

Adam achieves rapid convergence through adaptive learning rates and first/second-moment gradient estimation, reaching high validation accuracy with minimal oscillation.

#### AdamW

AdamW improves upon Adam by decoupling weight decay from gradient updates, producing stronger regularization and better generalization performance.

#### SGD with Momentum

Momentum accelerates optimization by incorporating previous gradient directions, helping the network escape shallow local minima and reducing noisy updates.

#### Vanilla SGD

Traditional SGD serves as a baseline optimizer and generally converges more slowly due to its lack of momentum and adaptive learning mechanisms.

---

# 🛠️ Project Structure

```text
├── data_loader.py
│   └── Data acquisition, normalization, and train-test splitting
│
├── inference.py
│   └── Standalone inference pipeline
│
├── main.py
│   └── Training loops, benchmarking, and hyperparameter configuration
│
├── best_iris_model.pkl
│   └── Serialized checkpoint containing the best-performing weights
│
└── model/
    ├── __init__.py
    ├── model.py
    │   └── Feedforward architecture and manual backpropagation
    │
    └── optimizers.py
        └── SGD, Momentum, Adam, and AdamW implementations
```

---

# 🧠 Mathematical Foundations

## 1. He (Kaiming) Initialization

To preserve variance across deep networks and maintain stable gradient flow, weights are initialized according to:

$$
W \sim \mathcal{N}\left(0,\frac{2}{n_{in}}\right)
$$

where:

$W$ = weight matrix
$n_{in}$ = number of incoming neurons to the layer

This initialization is particularly effective for ReLU-based networks.

---

## 2. Forward Propagation

For a hidden layer:

[
Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}
]

[
A^{[l]} = \text{ReLU}(Z^{[l]})
]

where:

[
\text{ReLU}(x)=\max(0,x)
]

For the output layer:

[
Z^{[L]} = A^{[L-1]}W^{[L]} + b^{[L]}
]

The Softmax function converts logits into class probabilities:

[
\hat{Y}*i=*
*\frac{e^{Z_i}}*
*{\sum*{j=1}^{K}e^{Z_j}}
]

where (K) is the number of classes.

---

## 3. Cross-Entropy Loss

For multi-class classification:

[
L=
-\frac{1}{m}
\sum_{i=1}^{m}
\sum_{c=1}^{K}
y_{ic}\log(\hat{y}_{ic})
]

where:

* (m) = batch size
* (K) = number of classes
* (y) = true labels
* (\hat{y}) = predicted probabilities

---

## 4. Backpropagation via Chain Rule

For Softmax combined with Cross-Entropy:

# \frac{\partial L}{\partial Z^{[L]}}

\hat{Y}-Y
]

Gradient of the final layer weights:

\frac{1}{m}
(A^{[L-1]})^T
\delta^{[L]}
]

Gradient of the final layer biases:

\frac{1}{m}
\sum \delta^{[L]}
]

Error propagated to a hidden layer:

(\delta^{[l+1]}(W^{[l+1]})^T)
\odot
\text{ReLU}'(Z^{[l]})
]

where:

[
\text{ReLU}'(z)=
\begin{cases}
1 & z > 0 \
0 & z \le 0
\end{cases}
]

Weight gradients:

\frac{1}{m}
(A^{[l-1]})^T
\delta^{[l]}
]

Bias gradients:

\frac{1}{m}
\sum \delta^{[l]}
]

---

## 5. Adam Optimization

Adam combines momentum and adaptive learning rates:

### First Moment Estimate

\beta_1 m_{t-1}
+
(1-\beta_1)g_t
]

### Second Moment Estimate

\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
]

### Bias Correction

\frac{m_t}{1-\beta_1^t}
]

\frac{v_t}{1-\beta_2^t}
]

### Parameter Update

## \theta_t

\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
]

---

## 6. AdamW Regularization

AdamW decouples weight decay from gradient estimation:

## \theta_t

\eta\lambda\theta_t
]

where:

* (\eta) = learning rate
* (\lambda) = weight decay coefficient
* (\epsilon) = numerical stability term

This formulation prevents weight decay from contaminating momentum statistics and typically improves generalization.

---

# 🏃‍♂️ Running the Project

## Install Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

---

## Train and Benchmark Optimizers

Execute the training suite:

```bash
python main.py
```

This will:

* Train all optimizer variants
* Generate convergence plots
* Evaluate validation performance
* Automatically save the best-performing model

---

## Run Production Inference

Load the serialized checkpoint and classify unseen samples:

```bash
python inference.py
```

This execution path performs prediction only and does not allocate memory for gradient tracking or training operations.

---

# 🌟 Strategic Takeaway: Transfer Learning Potential

Although originally trained on the Iris dataset, the architecture supports transfer learning workflows.

The learned feature extraction layers can be reused as a pretrained backbone in larger classification systems through:

### Feature Extraction

Freeze pretrained layers:

[
\frac{\partial L}{\partial W}=0
]

and train only newly added output layers.

### Fine-Tuning

Initialize a new model using pretrained weights and continue optimization with a smaller learning rate:

[
\eta_{\text{fine-tune}}
\ll
\eta_{\text{original}}
]

This allows the network to retain previously learned representations while adapting to new datasets and tasks.

---

## Summary

This project demonstrates a complete neural network implementation from first principles using only NumPy, covering:

* Forward propagation
* Backpropagation
* Cross-Entropy optimization
* He initialization
* SGD, Momentum, Adam, and AdamW optimizers
* Model checkpointing
* Production inference deployment
* Transfer learning foundations

The result is a fully transparent deep learning framework that exposes every mathematical operation involved in training modern neural networks.
