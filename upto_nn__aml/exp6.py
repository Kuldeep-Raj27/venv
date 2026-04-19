# ===================================================================
# EXPERIMENT 6: HIDDEN MARKOV MODEL (HMM)
# Objective: Analyze sequential data using probabilistic state transitions
#            and observations
# Algorithms: Viterbi (Decoding) + Forward (Likelihood Estimation)
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. CLASSIC EXAMPLE: Weather HMM
# -----------------------------
# Hidden States: 0 = Sunny, 1 = Rainy
# Observations:  0 = Dry (no umbrella), 1 = Wet (umbrella)

states = ['Sunny', 'Rainy']
n_states = len(states)

observations = ['Dry', 'Wet']
n_obs = len(observations)

print("Hidden States :", states)
print("Observations  :", observations, "\n")

# Transition Probability Matrix A: P(next_state | current_state)
# A[i][j] = probability of going from state i to state j
A = np.array([
    [0.8, 0.2],   # Sunny -> Sunny, Sunny -> Rainy
    [0.3, 0.7]    # Rainy -> Sunny, Rainy -> Rainy
])

# Emission Probability Matrix B: P(observation | state)
# B[i][k] = probability of observing k when in state i
B = np.array([
    [0.9, 0.1],   # Sunny: Dry=0.9, Wet=0.1
    [0.2, 0.8]    # Rainy: Dry=0.2, Wet=0.8
])

# Initial State Probabilities π
pi = np.array([0.6, 0.4])   # Start with Sunny=0.6, Rainy=0.4

# Example Observation Sequence (0=Dry, 1=Wet)
obs_sequence = [0, 1, 0, 0, 1, 1, 0]   # Dry, Wet, Dry, Dry, Wet, Wet, Dry
obs_names = [observations[o] for o in obs_sequence]

print("Observation Sequence:", obs_names)
print("Encoded as indices :", obs_sequence, "\n")

# -----------------------------
# 2. VITERBI ALGORITHM (Most Likely Hidden State Sequence)
# -----------------------------
def viterbi(obs, A, B, pi):
    T = len(obs)                    # length of observation sequence
    N = A.shape[0]                  # number of states
    
    # V[t][i] = probability of the most likely path ending in state i at time t
    V = np.zeros((T, N))
    # Backpointer to recover the path
    backpointer = np.zeros((T, N), dtype=int)
    
    # Initialization (t=0)
    V[0] = pi * B[:, obs[0]]
    backpointer[0] = 0
    
    # Recursion
    for t in range(1, T):
        for j in range(N):          # current state
            probs = V[t-1] * A[:, j] * B[j, obs[t]]
            V[t, j] = np.max(probs)
            backpointer[t, j] = np.argmax(probs)
    
    # Termination: best path probability and final state
    best_path_prob = np.max(V[-1])
    best_path_end_state = np.argmax(V[-1])
    
    # Backtrack to find the path
    best_path = [best_path_end_state]
    for t in range(T-1, 0, -1):
        best_path.append(backpointer[t, best_path[-1]])
    best_path.reverse()
    
    return best_path, best_path_prob, V

# Run Viterbi
path, path_prob, _ = viterbi(obs_sequence, A, B, pi)

print("=== VITERBI ALGORITHM (Decoding) ===")
print("Most likely hidden state sequence:")
print([states[s] for s in path])
print(f"Probability of this path: {path_prob:.6f}\n")

# -----------------------------
# 3. FORWARD ALGORITHM (Likelihood of Observation Sequence)
# -----------------------------
def forward(obs, A, B, pi):
    T = len(obs)
    N = A.shape[0]
    
    alpha = np.zeros((T, N))        # alpha[t][i] = P(O1..Ot, qt=Si | model)
    
    # Initialization
    alpha[0] = pi * B[:, obs[0]]
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]
    
    # Total likelihood P(O | model) = sum over all final states
    total_likelihood = np.sum(alpha[-1])
    
    return alpha, total_likelihood

# Run Forward
alpha, likelihood = forward(obs_sequence, A, B, pi)

print("=== FORWARD ALGORITHM (Likelihood Estimation) ===")
print(f"P(Observation Sequence | Model) = {likelihood:.8f}")
print("(This is the probability that the model generated the observed sequence)\n")

# -----------------------------
# 4. VISUALIZATION
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Observation Sequence + Decoded States
time_steps = range(len(obs_sequence))
axs[0].plot(time_steps, obs_sequence, 'o-', color='blue', label='Observations (0=Dry, 1=Wet)')
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['Dry', 'Wet'])
axs[0].set_title('Observation Sequence')
axs[0].grid(True, alpha=0.3)

# Plot decoded states on same axis (twin)
ax2 = axs[0].twinx()
ax2.step(time_steps, path, 's-', color='red', where='mid', label='Viterbi Decoded States')
ax2.set_yticks(range(n_states))
ax2.set_yticklabels(states)
ax2.set_ylabel('Hidden States (Viterbi)', color='red')

axs[0].legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 2: Forward Probabilities (alpha)
for i in range(n_states):
    axs[1].plot(time_steps, alpha[:, i], 'o-', label=f'P(state={states[i]} | observations so far)')
axs[1].set_title('Forward Probabilities α(t) over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Probability')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.suptitle('Hidden Markov Model - Weather Example\nViterbi Decoding + Forward Algorithm', fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. SUMMARY & KEY TAKEAWAYS
# -----------------------------
print("="*70)
print("KEY CONCEPTS LEARNED")
print("="*70)
print("• Hidden States     : Not directly observable (Sunny / Rainy)")
print("• Observations      : What we actually see (Dry / Wet)")
print("• Transition Matrix : Probability of moving between hidden states")
print("• Emission Matrix   : Probability of observing something given a state")
print("\n• Viterbi Algorithm : Finds the SINGLE most likely sequence of hidden states")
print("                      (Dynamic Programming - efficient)")
print("• Forward Algorithm : Computes the TOTAL probability of the observation sequence")
print("                      (Used for model evaluation / likelihood)")

print("\n✅ This experiment shows how HMMs are powerful for sequential data such as:")
print("   - Speech recognition, POS tagging, Gesture recognition")
print("   - Stock price trends, DNA sequences, Weather forecasting")