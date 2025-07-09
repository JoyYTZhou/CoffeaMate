import numpy as np

def calculate_rates(true_labels: np.ndarray, predicted_labels: np.ndarray):
    """Calculate FPR, FNR, TPR, TNR.

    Parameters:
    - true_labels: Boolean array where True represents a positive label.
    - predicted_labels: Boolean array where True represents a positive prediction.

    Returns:
    - Dictionary with FPR, FNR, TPR, TNR.
    """
    TP = np.sum((true_labels == True) & (predicted_labels == True))
    TN = np.sum((true_labels == False) & (predicted_labels == False))
    FP = np.sum((true_labels == False) & (predicted_labels == True))
    FN = np.sum((true_labels == True) & (predicted_labels == False))

    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {"FPR": FPR, "FNR": FNR, "TPR": TPR, "TNR": TNR}