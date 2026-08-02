import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_metrics(y_true, y_pred):
    """
    Calculates classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "confusion_matrix": cm.tolist()
    }

def plot_confusion_matrix(cm_list, save_path, title="Confusion Matrix"):
    """
    Plots and saves confusion matrix as an image.
    """
    cm = np.array(cm_list)
    plt.figure(figsize=(6, 5))
    
    # Custom colors and rich aesthetics
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Asli (0)', 'Counterfeit (1)'], 
        yticklabels=['Asli (0)', 'Counterfeit (1)'],
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")
