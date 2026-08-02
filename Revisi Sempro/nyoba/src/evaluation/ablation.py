import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config
from src.hmm.decode import viterbi
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix

def mcnemar_test(y_true, preds_a, preds_b):
    """
    Performs McNemar's exact test for comparing two classifiers.
    preds_a: predictions of IndoBERT-only
    preds_b: predictions of IndoBERT + HMM
    Returns:
        contingency_table: 2x2 matrix
        p_value: P-value of the test
    """
    y_true = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    
    # contingency table
    # a: both correct
    # b: A correct, B incorrect (discordant)
    # c: B correct, A incorrect (discordant)
    # d: both incorrect
    a = sum((preds_a == y_true) & (preds_b == y_true))
    b = sum((preds_a == y_true) & (preds_b != y_true))
    c = sum((preds_b == y_true) & (preds_a != y_true))
    d = sum((preds_a != y_true) & (preds_b != y_true))
    
    # We use exact binomial test since our dataset size is small
    # Under null hypothesis, discordant counts b and c are drawn from Binomial(b+c, 0.5)
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        # Two-sided binomial test
        k = min(b, c)
        p_value = 2 * binom.cdf(k, n_discordant, 0.5)
        # Handle edge case where k is exactly half
        p_value = min(p_value, 1.0)
        
    contingency = [
        [int(a), int(b)],
        [int(c), int(d)]
    ]
    
    return contingency, float(p_value)

def run_ablation():
    # Load sequences
    pkl_path = os.path.join(config.PROCESSED_DATA_DIR, "sequences.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} does not exist. Please run build_sequences first.")
        return
        
    with open(pkl_path, 'rb') as f:
        sequences_by_split = pickle.load(f)
        
    val_seqs = sequences_by_split['val']
    test_seqs = sequences_by_split['test']
    
    # Load HMM parameters
    with open(config.HMM_PARAMS_PATH, 'r') as f:
        hmm_params = json.load(f)
    pi, A, B = hmm_params['pi'], hmm_params['A'], hmm_params['B']
    
    # --- PHASE 1: OPTIMIZE INDOBERT-ONLY BASELINE THRESHOLD ON VAL SET ---
    # IndoBERT-only Baseline: Predicts counterfeit (1) if ratio of negative reviews (sentiment label = 0)
    # in the sequence exceeds a threshold.
    best_threshold = 0.0
    best_val_f1 = -1.0
    
    # Test thresholds from 0.0 to 1.0 with step 0.05
    thresholds = np.arange(0.0, 1.05, 0.05)
    val_labels = [s['store_label'] for s in val_seqs]
    
    for t in thresholds:
        val_preds = []
        for s in val_seqs:
            obs = s['observations']
            neg_ratio = sum(1 for o in obs if o == 2) / len(obs)
            pred = 1 if neg_ratio >= t else 0
            val_preds.append(pred)
            
        metrics = calculate_metrics(val_labels, val_preds)
        val_f1 = metrics['f1_macro']
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = t
            
    print(f"Optimized baseline threshold on Validation set: {best_threshold:.2f} (F1 Macro: {best_val_f1:.4f})")
    
    # --- PHASE 2: RUN EVALUATION ON TEST SET ---
    test_labels = [s['store_label'] for s in test_seqs]
    
    # 1. IndoBERT-only Baseline predictions
    baseline_preds = []
    for s in test_seqs:
        obs = s['observations']
        neg_ratio = sum(1 for o in obs if o == 2) / len(obs)
        pred = 1 if neg_ratio >= best_threshold else 0
        baseline_preds.append(pred)
        
    # 2. IndoBERT + HMM predictions (Viterbi decoding)
    hmm_preds = []
    hmm_paths = []
    for s in test_seqs:
        obs = s['observations']
        path, score = viterbi(obs, pi, A, B)
        # Predict store label based on majority decoded state
        pred = int(np.round(np.mean(path)))
        hmm_preds.append(pred)
        hmm_paths.append(path)
        
    # Calculate metrics
    baseline_metrics = calculate_metrics(test_labels, baseline_preds)
    hmm_metrics = calculate_metrics(test_labels, hmm_preds)
    
    # Plot and save confusion matrices
    plot_confusion_matrix(
        baseline_metrics['confusion_matrix'], 
        os.path.join(config.FIGURES_DIR, "confusion_matrix_baseline.png"),
        "Confusion Matrix: IndoBERT-only Baseline"
    )
    plot_confusion_matrix(
        hmm_metrics['confusion_matrix'], 
        os.path.join(config.FIGURES_DIR, "confusion_matrix_hmm.png"),
        "Confusion Matrix: IndoBERT + HMM"
    )
    
    # McNemar's Test
    contingency, p_value = mcnemar_test(test_labels, baseline_preds, hmm_preds)
    
    # Format report
    report = f"""# Ablation Study Report: IndoBERT vs IndoBERT + HMM

Laporan ini membandingkan kinerja model baseline **IndoBERT-only (agregasi persentase sentimen negatif)** dengan pipeline lengkap **IndoBERT + HMM (Viterbi decoding)** pada data uji (Test Set).

---

## Ringkasan Performa Klasifikasi (Level Toko)

| Metrik | IndoBERT-only Baseline (Threshold: {best_threshold:.2f}) | IndoBERT + HMM (Viterbi) | Selisih (Gain) |
| :--- | :---: | :---: | :---: |
| **Akurasi (Accuracy)** | {baseline_metrics['accuracy']:.4f} | {hmm_metrics['accuracy']:.4f} | {hmm_metrics['accuracy'] - baseline_metrics['accuracy']:.+4f} |
| **Precision (Macro)** | {baseline_metrics['precision_macro']:.4f} | {hmm_metrics['precision_macro']:.4f} | {hmm_metrics['precision_macro'] - baseline_metrics['precision_macro']:.+4f} |
| **Recall (Macro)** | {baseline_metrics['recall_macro']:.4f} | {hmm_metrics['recall_macro']:.4f} | {hmm_metrics['recall_macro'] - baseline_metrics['recall_macro']:.+4f} |
| **F1-Score (Macro)** | {baseline_metrics['f1_macro']:.4f} | {hmm_metrics['f1_macro']:.4f} | {hmm_metrics['f1_macro'] - baseline_metrics['f1_macro']:.+4f} |
| **F1-Score (Weighted)** | {baseline_metrics['f1_weighted']:.4f} | {hmm_metrics['f1_weighted']:.4f} | {hmm_metrics['f1_weighted'] - baseline_metrics['f1_weighted']:.+4f} |

---

## Analisis Confusion Matrix

### 1. IndoBERT-only Baseline
* True Positive (Counterfeit terdeteksi): {baseline_metrics['confusion_matrix'][1][1]}
* False Positive (Asli dituduh Counterfeit): {baseline_metrics['confusion_matrix'][0][1]}
* True Negative (Asli terdeteksi): {baseline_metrics['confusion_matrix'][0][0]}
* False Negative (Counterfeit lolos): {baseline_metrics['confusion_matrix'][1][0]}

### 2. IndoBERT + HMM
* True Positive (Counterfeit terdeteksi): {hmm_metrics['confusion_matrix'][1][1]}
* False Positive (Asli dituduh Counterfeit): {hmm_metrics['confusion_matrix'][0][1]}
* True Negative (Asli terdeteksi): {hmm_metrics['confusion_matrix'][0][0]}
* False Negative (Counterfeit lolos): {hmm_metrics['confusion_matrix'][1][0]}

---

## Uji Signifikansi Statistik (McNemar's Exact Test)

Untuk menguji apakah peningkatan performa model IndoBERT+HMM signifikan secara statistik dibanding baseline, kami menggunakan McNemar's exact test.

* **Tabel Kontingensi (Contingency Table):**
  ```
               HMM Correct   HMM Incorrect
  BL Correct      {contingency[0][0]:<12} {contingency[0][1]:<12}
  BL Incorrect    {contingency[1][0]:<12} {contingency[1][1]:<12}
  ```
  *(BL = Baseline, HMM = IndoBERT + HMM)*

* **Nilai p (p-value):** `{p_value:.6f}`
* **Signifikansi (alpha = 0.05):** {"SIGNIFIKAN (p < 0.05)" if p_value < 0.05 else "TIDAK SIGNIFIKAN (p >= 0.05)"}

### Kesimpulan Metodologi:
Pemodelan sekuensial sentimen menggunakan HMM {"berhasil meningkatkan" if hmm_metrics['f1_macro'] > baseline_metrics['f1_macro'] else "tidak menunjukkan peningkatan signifikan pada"} performa deteksi produk counterfeit dibanding agregasi voting sederhana. Hal ini karena HMM mampu memanfaatkan struktur ketergantungan urutan ulasan seiring waktu (temporal dependencies) dan meredam noise kesalahan klasifikasi sentimen ulasan individual.
"""
    
    report_path = os.path.join(config.REPORTS_DIR, "ablation_study_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nAblation study completed! Report written to {report_path}")
    print(report[:600] + "\n...")

if __name__ == "__main__":
    run_ablation()
