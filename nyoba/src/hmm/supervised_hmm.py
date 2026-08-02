import os
import sys
import pickle
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config

def estimate_hmm_params(alpha=None):
    if alpha is None:
        alpha = config.HMM_ALPHA
        
    pkl_path = os.path.join(config.PROCESSED_DATA_DIR, "sequences.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} does not exist. Please run build_sequences first.")
        return
        
    with open(pkl_path, 'rb') as f:
        sequences_by_split = pickle.load(f)
        
    train_seqs = sequences_by_split['train']
    print(f"Loaded {len(train_seqs)} training sequences for HMM parameter estimation.")
    
    # States: 0 = Asli, 1 = Counterfeit
    # Observations: 0 = Negative, 1 = Neutral, 2 = Positive (from IndoBERT)
    
    # 1. Initial State Probability (pi)
    pi_counts = np.zeros(2)
    for seq in train_seqs:
        label = seq['store_label']
        pi_counts[label] += 1
        
    # Apply Laplace smoothing
    pi = (pi_counts + alpha) / (sum(pi_counts) + 2 * alpha)
    
    # 2. Transition Matrix (A)
    # A[i][j] = P(S_t+1 = j | S_t = i)
    # Since store labels are static, state is constant for each sequence.
    # A sequence of length L has L-1 transitions of form state -> state.
    transition_counts = np.zeros((2, 2))
    for seq in train_seqs:
        label = seq['store_label']
        length = seq['length']
        if length > 1:
            transition_counts[label, label] += (length - 1)
            
    # Apply Laplace smoothing
    A = np.zeros((2, 2))
    for i in range(2):
        row_sum = sum(transition_counts[i])
        A[i] = (transition_counts[i] + alpha) / (row_sum + 2 * alpha)
        
    # 3. Emission Matrix (B)
    # B[i][k] = P(O_t = k | S_t = i)
    emission_counts = np.zeros((2, 3))
    for seq in train_seqs:
        label = seq['store_label']
        for obs in seq['observations']:
            emission_counts[label, obs] += 1
            
    # Apply Laplace smoothing
    B = np.zeros((2, 3))
    for i in range(2):
        row_sum = sum(emission_counts[i])
        B[i] = (emission_counts[i] + alpha) / (row_sum + 3 * alpha)
        
    # Save HMM params
    params = {
        "pi": pi.tolist(),
        "A": A.tolist(),
        "B": B.tolist()
    }
    
    with open(config.HMM_PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=4)
        
    print(f"HMM parameters successfully saved to {config.HMM_PARAMS_PATH}")
    print("\nInitial State Distribution (pi):")
    print(f"  Asli: {pi[0]:.4f}")
    print(f"  Counterfeit: {pi[1]:.4f}")
    
    print("\nTransition Matrix (A):")
    print(f"  From Asli: Asli={A[0,0]:.6f}, Counterfeit={A[0,1]:.6f}")
    print(f"  From Counterfeit: Asli={A[1,0]:.6f}, Counterfeit={A[1,1]:.6f}")
    
    print("\nEmission Matrix (B) - P(Sentiment | State):")
    print(f"  Asli State: Neg={B[0,0]:.4f}, Neu={B[0,1]:.4f}, Pos={B[0,2]:.4f}")
    print(f"  Counterfeit State: Neg={B[1,0]:.4f}, Neu={B[1,1]:.4f}, Pos={B[1,2]:.4f}")

if __name__ == "__main__":
    estimate_hmm_params()
