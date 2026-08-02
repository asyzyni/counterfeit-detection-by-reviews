import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config

def viterbi(obs_seq, pi, A, B):
    """
    Implements the standard Viterbi algorithm in log-space.
    Args:
        obs_seq: List of observation indices (0, 1, 2)
        pi: Initial state distribution (length 2)
        A: State transition matrix (2x2)
        B: Emission probability matrix (2x3)
    Returns:
        best_path: List of decoded hidden states (0 or 1)
        best_score: Log probability of the best path
    """
    T = len(obs_seq)
    N = 2 # 2 states (0: Asli, 1: Counterfeit)
    
    # Avoid log(0) using a tiny offset
    epsilon = 1e-15
    log_pi = np.log(np.array(pi) + epsilon)
    log_A = np.log(np.array(A) + epsilon)
    log_B = np.log(np.array(B) + epsilon)
    
    # Trellis matrices
    # delta[t, i] stores the max log prob of path ending at state i at time t
    delta = np.zeros((T, N))
    # psi[t, i] stores the backpointer to the state at t-1
    psi = np.zeros((T, N), dtype=int)
    
    # Initialization
    first_obs = obs_seq[0]
    for i in range(N):
        delta[0, i] = log_pi[i] + log_B[i, first_obs]
        
    # Recursion
    for t in range(1, T):
        obs = obs_seq[t]
        for j in range(N):
            # Calculate path probability from all previous states i to current state j
            probs = delta[t-1, :] + log_A[:, j]
            best_prev_state = np.argmax(probs)
            delta[t, j] = probs[best_prev_state] + log_B[j, obs]
            psi[t, j] = best_prev_state
            
    # Path termination
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(delta[T-1, :])
    best_score = delta[T-1, best_path[T-1]]
    
    # Backtracking
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]
        
    return best_path.tolist(), float(best_score)

def decode_sequence(obs_seq):
    """
    Decodes an observation sequence using stored HMM parameters.
    """
    if not os.path.exists(config.HMM_PARAMS_PATH):
        raise FileNotFoundError(f"HMM parameters file not found at {config.HMM_PARAMS_PATH}. Please train the HMM first.")
        
    with open(config.HMM_PARAMS_PATH, 'r') as f:
        params = json.load(f)
        
    pi = params['pi']
    A = params['A']
    B = params['B']
    
    path, score = viterbi(obs_seq, pi, A, B)
    
    # The predicted store label is the majority state in the decoded path
    pred_label = int(np.round(np.mean(path)))
    
    return pred_label, path, score

if __name__ == "__main__":
    # Quick sanity check with arbitrary sequence
    test_pi = [0.5, 0.5]
    test_A = [[0.9, 0.1], [0.1, 0.9]]
    # Positive sentiment [2] should be emitted more by Asli [0]
    # Negative sentiment [0] should be emitted more by Counterfeit [1]
    test_B = [[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]]
    
    # A sequence with mostly positive reviews [2, 2, 2, 1, 2]
    path_asli, score_asli = viterbi([2, 2, 2, 1, 2], test_pi, test_A, test_B)
    print("Test positive sequence decoded path:", path_asli, "Score:", score_asli)
    
    # A sequence with mostly negative reviews [0, 0, 0, 1, 0]
    path_counterfeit, score_counterfeit = viterbi([0, 0, 0, 1, 0], test_pi, test_A, test_B)
    print("Test negative sequence decoded path:", path_counterfeit, "Score:", score_counterfeit)
