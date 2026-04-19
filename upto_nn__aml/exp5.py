import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Train EM model (Gaussian Mixture)
gmm = GaussianMixture(n_components=3)

gmm.fit(X)

# Predict cluster probabilities
labels = gmm.predict(X)

# Plot clusters
plt.scatter(X[:,0], X[:,1], c=labels)

# Plot cluster centers
centers = gmm.means_

plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X')

plt.title("Clustering using EM Algorithm")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()# ===================================================================
# EXPERIMENT: EXPECTATION-MAXIMIZATION (EM) ALGORITHM
# Gaussian Mixture Model (GMM) from Scratch
# Objective: Learn probabilistic clustering using EM
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# -----------------------------
# 1. GENERATE SYNTHETIC DATA (3 overlapping Gaussians)
# -----------------------------
def generate_gmm_data(n_samples=500):
    # True parameters (hidden from the model)
    means_true = np.array([[-2, -2], [2, 2], [0, 4]])
    covs_true = np.array([[[1, 0.5], [0.5, 1]],
                          [[1, -0.5], [-0.5, 1]],
                          [[1, 0], [0, 1]]])
    
    # Generate data
    X = []
    true_labels = []
    for i in range(3):
        cluster = np.random.multivariate_normal(means_true[i], covs_true[i], n_samples//3)
        X.append(cluster)
        true_labels.append(np.full(n_samples//3, i))
    
    X = np.vstack(X)
    true_labels = np.hstack(true_labels)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], true_labels[indices]

X, true_labels = generate_gmm_data(n_samples=600)

print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

# -----------------------------
# 2. GMM + EM ALGORITHM IMPLEMENTATION
# -----------------------------
class GaussianMixtureEM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        n, d = X.shape
        self.n, self.d = n, d
        
        # 1. Initialize parameters
        # Randomly choose k points as initial means
        idx = np.random.choice(n, self.k, replace=False)
        self.means = X[idx]
        
        # Initial covariances (identity)
        self.covs = np.array([np.eye(d) for _ in range(self.k)])
        
        # Initial mixing coefficients (equal)
        self.pis = np.ones(self.k) / self.k
        
        log_likelihoods = []
        
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities (posterior probabilities)
            responsibilities = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_lik = self._log_likelihood(X)
            log_likelihoods.append(log_lik)
            
            # Check convergence
            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                print(f"Converged after {iteration+1} iterations")
                break
                
        self.log_likelihoods = log_likelihoods
        return self
    
    def _e_step(self, X):
        """Expectation step: compute responsibility of each component"""
        n = X.shape[0]
        resp = np.zeros((n, self.k))
        
        for j in range(self.k):
            # Multivariate Gaussian PDF
            pdf = multivariate_normal.pdf(X, mean=self.means[j], cov=self.covs[j])
            resp[:, j] = self.pis[j] * pdf
        
        # Normalize so that sum over components = 1 for each point
        resp /= resp.sum(axis=1, keepdims=True)
        return resp
    
    def _m_step(self, X, responsibilities):
        """Maximization step: update means, covariances, and pis"""
        n = X.shape[0]
        Nk = responsibilities.sum(axis=0)  # effective number of points per component
        
        # Update means
        for j in range(self.k):
            self.means[j] = (responsibilities[:, j][:, np.newaxis] * X).sum(axis=0) / Nk[j]
        
        # Update covariances
        for j in range(self.k):
            diff = X - self.means[j]
            self.covs[j] = (responsibilities[:, j][:, np.newaxis] * diff).T @ diff / Nk[j]
            # Add small diagonal for numerical stability
            self.covs[j] += 1e-6 * np.eye(self.d)
        
        # Update mixing coefficients
        self.pis = Nk / n
    
    def _log_likelihood(self, X):
        """Compute current log-likelihood"""
        n = X.shape[0]
        ll = 0
        for j in range(self.k):
            pdf = multivariate_normal.pdf(X, mean=self.means[j], cov=self.covs[j])
            ll += self.pis[j] * pdf
        return np.sum(np.log(ll + 1e-10))
    
    def predict_proba(self, X):
        """Return soft assignments (responsibilities)"""
        return self._e_step(X)
    
    def predict(self, X):
        """Hard assignment: component with highest responsibility"""
        return self.predict_proba(X).argmax(axis=1)


# -----------------------------
# 3. TRAIN THE MODEL
# -----------------------------
gmm = GaussianMixtureEM(n_components=3, max_iter=200, tol=1e-5)
gmm.fit(X)

print(f"Final mixing coefficients (π): {np.round(gmm.pis, 3)}")

# -----------------------------
# 4. VISUALIZATION
# -----------------------------
plt.figure(figsize=(15, 5))

# Plot 1: Data with true labels
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', s=30, alpha=0.7)
plt.title("Original Data (True Labels)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot 2: GMM soft clustering (responsibilities)
plt.subplot(1, 3, 2)
probs = gmm.predict_proba(X)
# Color by dominant component + intensity by confidence
colors = probs @ np.array([[1,0,0], [0,1,0], [0,0,1]])  # RGB mixing
plt.scatter(X[:, 0], X[:, 1], c=colors, s=30, alpha=0.8)
plt.title("GMM Soft Clustering\n(EM Algorithm)")
plt.xlabel("Feature 1")

# Plot 3: Hard assignments + learned means
plt.subplot(1, 3, 3)
pred_labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis', s=30, alpha=0.7)
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', s=200, marker='X', 
            edgecolors='black', label='Learned Means')
plt.title("GMM Hard Clustering + Learned Means")
plt.xlabel("Feature 1")
plt.legend()

plt.suptitle("Expectation-Maximization for Gaussian Mixture Model", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. LOG-LIKELIHOOD CONVERGENCE
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(gmm.log_likelihoods, 'b-', linewidth=2)
plt.title("EM Algorithm Convergence: Log-Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True, alpha=0.3)
plt.show()

print("\n✅ Key Takeaways from EM for GMM:")
print("• E-step → Computes 'soft' assignments (probabilities)")
print("• M-step → Updates means, covariances, and mixing weights")
print("• EM iteratively increases the likelihood until convergence")
print("• GMM provides probabilistic clustering (unlike hard K-Means)")