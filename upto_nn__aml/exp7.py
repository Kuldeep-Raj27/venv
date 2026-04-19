# ===================================================================
# EXPERIMENT 7: NEURAL NETWORK FROM SCRATCH (Multi-Layer Perceptron)
# Objective: Understand forward pass, backpropagation (chain rule),
#            and how a neural net learns non-linear patterns like XOR
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. XOR DATASET (Non-linearly separable)
# -----------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

print("Input X:\n", X)
print("Target y (XOR):\n", y, "\n")

# -----------------------------
# 2. SIGMOID ACTIVATION & DERIVATIVE
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid: a * (1 - a)"""
    return a * (1 - a)

# -----------------------------
# 3. NEURAL NETWORK CLASS (Manual Forward + Backprop)
# -----------------------------
class MLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.5):
        self.lr = lr
        
        # Initialize weights and biases (small random values)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5   # (2, 4)
        self.b1 = np.zeros((1, hidden_size))                       # (1, 4)
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5  # (4, 1)
        self.b2 = np.zeros((1, output_size))                       # (1, 1)
        
        self.loss_history = []
    
    def forward(self, X):
        """Forward Pass"""
        self.z1 = X @ self.W1 + self.b1          # Linear: hidden pre-activation
        self.a1 = sigmoid(self.z1)               # Hidden activation
        
        self.z2 = self.a1 @ self.W2 + self.b2    # Linear: output pre-activation
        self.a2 = sigmoid(self.z2)               # Output (prediction)
        
        return self.a2
    
    def backward(self, X, y_true, y_pred):
        """Backward Pass using Chain Rule (Backpropagation)"""
        m = X.shape[0]   # number of samples
        
        # ------------------- Output Layer Error -------------------
        # dL/dz2 = (y_pred - y_true) * sigmoid'(z2)
        dz2 = (y_pred - y_true) * sigmoid_derivative(y_pred)   # (4, 1)
        
        # Gradients for W2 and b2
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # ------------------- Hidden Layer Error -------------------
        # Propagate error back: dz1 = dz2 * W2^T * sigmoid'(z1)
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.a1)   # (4, 4)
        
        # Gradients for W1 and b1
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # ------------------- Update Weights (Gradient Descent) -------------------
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute Mean Squared Error loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, y_pred)
            
            # Print progress every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.6f}")
        
        print("\nTraining completed!\n")
    
    def predict(self, X):
        """Return binary predictions (0 or 1)"""
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)


# -----------------------------
# 4. TRAIN THE NETWORK
# -----------------------------
model = MLP(input_size=2, hidden_size=4, output_size=1, lr=0.8)

print("Training Neural Network on XOR problem...\n")
model.train(X, y, epochs=8000)

# -----------------------------
# 5. EVALUATION
# -----------------------------
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y) * 100

print("Final Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Predicted: {y_pred[i][0]} | True: {y[i][0]}")

print(f"\nTraining Accuracy: {accuracy:.2f}%")

# -----------------------------
# 6. LOSS CURVE
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(model.loss_history, color='blue', linewidth=2)
plt.title('Training Loss (Mean Squared Error) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 7. DECISION BOUNDARY VISUALIZATION (Optional but insightful)
# -----------------------------
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100),
                     np.linspace(-0.2, 1.2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.forward(grid)
Z = (Z > 0.5).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', s=100)
plt.title('Decision Boundary Learned by MLP for XOR')
plt.xlabel('Input Feature 1')
plt.ylabel('Input Feature 2')
plt.grid(True, alpha=0.3)
plt.show()