# ===================================================================
# EXPERIMENT 3: SUPPORT VECTOR MACHINE (SVM)
# Objective: Understand how SVM finds the optimal hyperplane 
#            that separates classes with maximum margin
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. LOAD AND PREPARE DATA
# -----------------------------
iris = datasets.load_iris()

# Use only two classes for clear binary classification (Setosa vs Versicolor)
# and only 2 features for easy 2D visualization
X = iris.data[:100, :2]      # sepal length & sepal width
y = iris.target[:100]        # 0 = Setosa, 1 = Versicolor

# Feature scaling (very important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]} (Sepal Length, Sepal Width)\n")

# -----------------------------
# 2. TRAIN LINEAR SVM
# -----------------------------
model = SVC(kernel='linear', C=1.0)   # C = regularization parameter
model.fit(X_train, y_train)

# -----------------------------
# 3. EVALUATION
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== Model Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor']))

# Support vectors information
print(f"\nNumber of Support Vectors: {len(model.support_vectors_)}")
print(f"Support vectors per class: {model.n_support_}")

# -----------------------------
# 4. VISUALIZE DECISION BOUNDARY (Most Important Part)
# -----------------------------
plt.figure(figsize=(10, 7))

# Plot training points
plt.scatter(X_train[:, 0], X_train[:, 1], 
            c=y_train, cmap='bwr', edgecolors='k', 
            s=80, label='Training data')

# Plot test points
plt.scatter(X_test[:, 0], X_test[:, 1], 
            c=y_test, cmap='bwr', edgecolors='k', 
            marker='^', s=100, label='Test data')

# Plot decision boundary
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0])

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='yellow', 
            linewidth=2, label='Support Vectors')

plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('Support Vector Machine (Linear Kernel)\nMaximum Margin Hyperplane')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 5. Effect of C (Regularization Parameter)
# -----------------------------
print("\n" + "="*60)
print("Effect of C parameter (try different values):")
print("="*60)

for C in [0.1, 1, 10, 100]:
    model_c = SVC(kernel='linear', C=C)
    model_c.fit(X_train, y_train)
    acc = accuracy_score(y_test, model_c.predict(X_test))
    sv_count = len(model_c.support_vectors_)
    print(f"C = {C:6} → Accuracy: {acc:.4f} | Support Vectors: {sv_count}")