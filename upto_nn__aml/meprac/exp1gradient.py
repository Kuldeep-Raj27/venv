import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression

# -----------------------------
# DATA
# -----------------------------
X, y = make_regression(n_samples=4, n_features=1, noise=80, random_state=13)

X_log, y_log = make_classification(
    n_samples=20, n_features=1, n_informative=1,
    n_redundant=0, n_clusters_per_class=1, random_state=13
)

# OLS (baseline for linear)
reg = LinearRegression().fit(X, y)


# -----------------------------
# HELPER
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------
# GENERIC GD FUNCTION
# -----------------------------
def GD(X, y, model="linear", method="batch", lr=0.01, epochs=20, batch_size=2):
    
    m, b = 0, 0
    history = []
    
    X = X.ravel()
    y = y.ravel()
    n = len(X)
    
    for epoch in range(epochs):
        
        # shuffle for sgd/mini
        if method != "batch":
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
        else:
            X_s, y_s = X, y
        
        # batching logic
        if method == "batch":
            batches = [(X_s, y_s)]
        elif method == "mini":
            batches = [(X_s[i:i+batch_size], y_s[i:i+batch_size]) for i in range(0, n, batch_size)]
        else:  # sgd
            batches = [(X_s[i:i+1], y_s[i:i+1]) for i in range(n)]
        
        for X_b, y_b in batches:
            
            z = m * X_b + b
            
            if model == "linear":
                y_pred = z
                dm = (-2/len(X_b)) * np.sum(X_b * (y_b - y_pred))
                db = (-2/len(X_b)) * np.sum(y_b - y_pred)
            
            else:  # logistic
                y_pred = sigmoid(z)
                dm = (1/len(X_b)) * np.sum((y_pred - y_b) * X_b)
                db = (1/len(X_b)) * np.sum(y_pred - y_b)
            
            m -= lr * dm
            b -= lr * db
        
        history.append((m, b))
    
    return history


# -----------------------------
# RUN ALL
# -----------------------------
batch_lin = GD(X, y, "linear", "batch")
mini_lin  = GD(X, y, "linear", "mini")
sgd_lin   = GD(X, y, "linear", "sgd")

batch_log = GD(X_log, y_log, "logistic", "batch", lr=0.1)
mini_log  = GD(X_log, y_log, "logistic", "mini", lr=0.1)
sgd_log   = GD(X_log, y_log, "logistic", "sgd", lr=0.1)


# -----------------------------
# PLOT LINEAR
# -----------------------------
X_line = np.linspace(X.min(), X.max(), 100)

plt.figure()
plt.scatter(X, y)

# OLS
plt.plot(X_line, reg.coef_[0]*X_line + reg.intercept_, label="OLS", linewidth=3)

# GD results
for hist, label in zip([batch_lin, mini_lin, sgd_lin], ["Batch", "Mini", "SGD"]):
    m, b = hist[-1]
    plt.plot(X_line, m*X_line + b, label=label)

plt.title("Linear Regression")
plt.legend()
plt.show()


# -----------------------------
# PLOT LOGISTIC
# -----------------------------
plt.figure()
plt.scatter(X_log, y_log)

X_line = np.linspace(X_log.min(), X_log.max(), 100)

for hist, label in zip([batch_log, mini_log, sgd_log], ["Batch", "Mini", "SGD"]):
    m, b = hist[-1]
    y_prob = sigmoid(m * X_line + b)
    plt.plot(X_line, y_prob, label=label)

plt.title("Logistic Regression")
plt.legend()
plt.show()


# -----------------------------
# TRAJECTORY PLOT
# -----------------------------
def plot_traj(hist, label):
    m_vals = [h[0] for h in hist]
    b_vals = [h[1] for h in hist]
    plt.plot(m_vals, b_vals, marker='o', label=label)

plt.figure()
plot_traj(batch_lin, "Batch Linear")
plot_traj(mini_lin, "Mini Linear")
plot_traj(sgd_lin, "SGD Linear")
plt.title("Trajectory (Linear)")
plt.xlabel("m")
plt.ylabel("b")
plt.legend()
plt.show()

plt.figure()
plot_traj(batch_log, "Batch Logistic")
plot_traj(mini_log, "Mini Logistic")
plot_traj(sgd_log, "SGD Logistic")
plt.title("Trajectory (Logistic)")
plt.xlabel("m")
plt.ylabel("b")
plt.legend()
plt.show()