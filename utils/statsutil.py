import numpy as np
import logging
import pandas as pd

def normalize_mc(data_df, mc_df, feature='DiTau_mass') -> tuple[pd.DataFrame, float]:
    from src.utils.plotutil import HistogramHelper

    renorm_fac = HistogramHelper.get_normalization_factor(mc_df[feature], data_df[feature], bins=30, range=(0, 300), 
                                weights_a=mc_df['weight'], weights_b=data_df['weight'])
    mc_df['weight'] *= renorm_fac
    logging.info(f"MC normalized by factor {renorm_fac:.4f} to match Data in {feature} distribution")
    return mc_df, renorm_fac

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