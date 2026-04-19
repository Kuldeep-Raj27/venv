# ===================================================================
# EXPERIMENT 4: THE KERNEL TRICK IN SUPPORT VECTOR MACHINES
# Objective: Understand how kernels (Linear, Polynomial, RBF) 
#            solve non-linearly separable problems
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. GENERATE NON-LINEAR DATASETS
# -----------------------------
np.random.seed(42)

# Dataset 1: Concentric Circles (classic kernel example)
X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.1, random_state=42)

# Dataset 2: Two Moons (another good non-linear example)
X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)

# We'll use Circles dataset for main demonstration (you can switch easily)
X, y = X_circles, y_circles
dataset_name = "Circles Dataset"

# Optional: Uncomment to use Moons instead
# X, y = X_moons, y_moons
# dataset_name = "Moons Dataset"

# Feature scaling (highly recommended for SVM with kernels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Dataset: {dataset_name}")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")

# -----------------------------
# 2. DEFINE KERNELS TO COMPARE
# -----------------------------
kernels = ['linear', 'poly', 'rbf']
kernel_names = ['Linear Kernel', 'Polynomial Kernel (degree=3)', 'RBF (Gaussian) Kernel']

# -----------------------------
# 3. TRAIN & VISUALIZE FOR EACH KERNEL
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Kernel Trick in SVM - {dataset_name}\n'
             'How different kernels create non-linear decision boundaries', 
             fontsize=16)

for i, kernel in enumerate(kernels):
    # Train SVM with current kernel
    if kernel == 'poly':
        model = SVC(kernel=kernel, degree=3, C=1.0, gamma='scale')
    else:
        model = SVC(kernel=kernel, C=1.0, gamma='scale')
    
    model.fit(X_train, y_train)
    
    # Accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Plot decision boundary
    ax = axes[i]
    
    # Create mesh grid
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    ax.contour(xx, yy, Z, colors='black', linewidths=1.5, levels=[0])
    
    # Plot training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
               cmap='bwr', edgecolors='k', s=60, label='Training data')
    
    # Highlight support vectors
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none', 
               edgecolors='yellow', linewidth=2, label='Support Vectors')
    
    ax.set_title(f'{kernel_names[i]}\n'
                 f'Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}\n'
                 f'Support Vectors: {len(sv)}')
    ax.set_xlabel('Feature 1 (scaled)')
    ax.set_ylabel('Feature 2 (scaled)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# -----------------------------
# 4. SUMMARY & OBSERVATIONS
# -----------------------------
print("\n" + "="*70)
print("KERNEL TRICK SUMMARY")
print("="*70)

for i, kernel in enumerate(kernels):
    if kernel == 'poly':
        model = SVC(kernel=kernel, degree=3, C=1.0, gamma='scale')
    else:
        model = SVC(kernel=kernel, C=1.0, gamma='scale')
    
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    sv_count = len(model.support_vectors_)
    
    print(f"{kernel_names[i]:30} → Test Accuracy: {test_acc:.4f} | "
          f"Support Vectors: {sv_count}")

print("\n✅ Key Takeaways:")
print("• Linear kernel   → Cannot separate circles/moons (straight line only)")
print("• Polynomial kernel → Can create curved boundaries (higher degree = more complex)")
print("• RBF kernel      → Most powerful for complex non-linear shapes")
print("• Kernel Trick allows SVM to work in higher dimensions without explicitly computing them!")