#Batch gradient descent

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
np.random.seed(42)

# For Linear Regression
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# -----------------------------
# 2. Cost Function (MSE)
# -----------------------------
def compute_cost(X, y, theta):
    m = len(y)
    return (1/m) * np.sum((X.dot(theta) - y) ** 2)

# -----------------------------
# 3. Batch Gradient Descent
# -----------------------------
def batch_gradient_descent(X, y, lr=0.1, iterations=50):
    theta = np.random.randn(2, 1)
    theta_path = [theta.copy()]
    cost_history = []

    for _ in range(iterations):
        gradients = 2/len(y) * X.T.dot(X.dot(theta) - y)
        theta = theta - lr * gradients
        theta_path.append(theta.copy())
        cost_history.append(compute_cost(X, y, theta))

    return theta, theta_path, cost_history

# -----------------------------
# 4. Stochastic Gradient Descent
# -----------------------------
def stochastic_gradient_descent(X, y, lr=0.01, epochs=20):
    theta = np.random.randn(2, 1)
    theta_path = [theta.copy()]
    cost_history = []

    for _ in range(epochs):
        for i in range(len(y)):
            random_index = np.random.randint(len(y))
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - lr * gradients

        theta_path.append(theta.copy())
        cost_history.append(compute_cost(X, y, theta))

    return theta, theta_path, cost_history

# -----------------------------
# 5. Mini-Batch Gradient Descent
# -----------------------------
def mini_batch_gradient_descent(X, y, lr=0.05, iterations=50, batch_size=20):
    theta = np.random.randn(2, 1)
    theta_path = [theta.copy()]
    cost_history = []
    m = len(y)

    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            gradients = 2/len(yi) * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - lr * gradients

        theta_path.append(theta.copy())
        cost_history.append(compute_cost(X, y, theta))

    return theta, theta_path, cost_history

# -----------------------------
# 6. Run Gradient Descent Variants
# -----------------------------
theta_b, path_b, cost_b = batch_gradient_descent(X_b, y)
theta_s, path_s, cost_s = stochastic_gradient_descent(X_b, y)
theta_m, path_m, cost_m = mini_batch_gradient_descent(X_b, y)

# -----------------------------
# 7. Contour Plot of Cost Function
# -----------------------------
theta0_vals = np.linspace(-2, 6, 100)
theta1_vals = np.linspace(0, 6, 100)
J_vals = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        theta = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = compute_cost(X_b, y, theta)

plt.figure(figsize=(8, 6))
plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=30)
plt.xlabel("θ₀")
plt.ylabel("θ₁")
plt.title("Cost Function Contours")

# -----------------------------
# 8. Plot Weight Trajectories
# -----------------------------
def plot_path(path, label, style):
    theta0 = [t[0][0] for t in path]
    theta1 = [t[1][0] for t in path]
    plt.plot(theta0, theta1, style, label=label)

plot_path(path_b, "Batch GD", "r-o")
plot_path(path_s, "SGD", "g-x")
plot_path(path_m, "Mini-Batch GD", "b-s")

plt.legend()
plt.show()

# -----------------------------
# 9. Cost vs Iterations Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(cost_b, label="Batch GD")
plt.plot(cost_s, label="SGD")
plt.plot(cost_m, label="Mini-Batch GD")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence Comparison")
plt.legend()
plt.show()

# -----------------------------
# 10. Logistic Regression (Basic)
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def logistic_gd(X, y, lr=0.1, iterations=100):
    theta = np.zeros((X.shape[1], 1))
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradients = (1/len(y)) * X.T.dot(h - y)
        theta -= lr * gradients
        cost_history.append(log_loss(X, y, theta))

    return theta, cost_history

print("Experiment 1 completed successfully.")
