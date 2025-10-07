import os, logging
import torch
import torch.nn as nn
import numpy as np

def data_subtract_mc(total_df):
    """Subtract MC from total data by inverting the sign of MC weights."""
    copy = total_df.copy()
    copy.loc[copy['group'] != 'Data', 'weight'] *= -1
    return copy

def infer_multiclass(model, data_df_0, features):
    """
    Standalone function to perform multiclass inference and compute reweights.
    
    Args:
        model: Trained multiclass model
        data_df_0: DataFrame containing the data to reweight
        features: List of feature names to use for inference
        
    Returns:
        dict: Contains model predictions and computed weights
        'w_reco_qcd' is unnormalized to the total number of predicted events in the target region.
    """
    # Prediction on DATA From SS only
    X_data = data_df_0[features].to_numpy().astype(np.float32)
    X_data_tensor = torch.from_numpy(X_data)
    
    with torch.no_grad():
        logits = model(X_data_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    s_data_0 = probs[:, 0]  # Probability for class 0
    s_mc = probs[:, 1]      # Probability for class 1
    s_data_1 = probs[:, 2]  # Probability for class 2
    
    logging.info(f"Max probability for class 0 (Data in original region): {s_data_0.max():.4f}")
    logging.info(f"Min probability for class 0 (Data in original region): {s_data_0.min():.4f}")
    logging.info(f"Max probability for class 1 (MC): {s_mc.max():.4f}")
    logging.info(f"Min probability for class 1 (MC): {s_mc.min():.4f}")
    logging.info(f"Max probability for class 2 (Data in new region): {s_data_1.max():.4f}")
    logging.info(f"Min probability for class 2 (Data in new region): {s_data_1.min():.4f}")
    
    # Compute reweight
    weights = data_df_0["weight"].to_numpy() if "weight" in data_df_0.columns else np.ones(len(data_df_0))
    w_reco_qcd = np.maximum(s_data_1 - s_mc, 0) / (s_data_0 + 1e-7) * weights

    results_dict = {
        "w_reco_qcd": w_reco_qcd,
        "s_data_0": s_data_0,
        "s_mc": s_mc,
        "s_data_1": s_data_1
    }
    
    return results_dict

def infer_data_to_mc(model, data_df, mc_df, features):
    """
    Perform inference using a trained Data vs MC classifier.
    
    Args:
        model: Trained PyTorch model
        features: List of feature column names
        
    Returns:
        dict with inference results
    """
    # Prepare data features
    p1 = data_df[features].to_numpy().astype(np.float32)   # Data
    
    # 2) Prediction on DATA only
    X_data_tensor = torch.from_numpy(p1)
    with torch.no_grad():
        s_data = torch.sigmoid(model(X_data_tensor)).numpy().ravel()

    eps = 1e-6
    r_data = s_data / (1.0 - s_data + eps)   # local ratio p2/p1

    logging.info(f"Max probability for belonging to MC only: {s_data.max():.4f}")
    logging.info(f"Min probability for belonging to MC only: {s_data.min():.4f}")
    logging.info(f"Max ratio of QCD/Data: {r_data.max():.4f}")
    logging.info(f"Min ratio of QCD/Data: {r_data.min():.4f}")

    # Base weights from data
    w_data_base = data_df["weight"].to_numpy() if "weight" in data_df.columns else np.ones(len(p1), dtype=np.float32)
    w_data_reco_p3 = (1.0 - r_data) * w_data_base
    w_data_reco_p2 = r_data * w_data_base

    # -----------------------
    # 4) Normalization
    # -----------------------
    target_sum = data_df["weight"].sum() - mc_df['weight'].sum()
    norm_factor_p3 = target_sum / (w_data_reco_p3.sum() + 1e-12)
    norm_factor_p2 = target_sum / (w_data_reco_p2.sum() + 1e-12)

    w_data_reco_p3 *= norm_factor_p3
    w_data_reco_p2 *= norm_factor_p2

    return {
        "w_data_reco_p3": w_data_reco_p3,
        "w_data_reco_p2": w_data_reco_p2
    }
    
def infer_ss_to_os(model, ss_df, os_df, features):
    """
    Use a trained SS/OS classifier to compute OS weights for SS events.
    
    Args:
        model: trained SimpleNN model
        ss_df (pd.DataFrame): Same-sign events
        os_df (pd.DataFrame): Opposite-sign events (for normalization)
        features (list): list of feature column names
    
    Returns:
        dict with inference results
    """
    # Prepare SS data for inference
    p1 = ss_df[features].to_numpy().astype(np.float32)
    
    # Prediction on SS only
    X_ss_tensor = torch.from_numpy(p1)
    with torch.no_grad():
        s_data = torch.sigmoid(model(X_ss_tensor)).numpy().ravel()

    eps = 1e-6
    r_data = s_data / (1.0 - s_data + eps)   # local ratio OS/SS

    logging.info(f"Max probability for OS: {s_data.max():.4f}")
    logging.info(f"Min probability for OS: {s_data.min():.4f}")
    logging.info(f"Max ratio of OS/SS: {r_data.max():.4f}")
    logging.info(f"Min ratio of OS/SS: {r_data.min():.4f}")

    # Compute OS weights for SS events
    w_data_base = ss_df["weight"].to_numpy() if "weight" in ss_df.columns else np.ones(len(p1), dtype=np.float32)
    w_data_reco_OS = r_data * w_data_base

    # Normalize to match OS total weight
    target_sum = os_df["weight"].sum()
    norm_factor = target_sum / (w_data_reco_OS.sum() + 1e-12)
    w_data_reco_OS *= norm_factor

    return {
        "r_data": r_data,
        "w_data_reco_os": w_data_reco_OS
    }
