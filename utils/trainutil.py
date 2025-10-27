
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from src.utils.inferutil import infer_multiclass, infer_data_to_mc

def smooth_labels(y, eps=0.01):
    """Smoothing of binary labels. eps belonging to [0.01, 0.1] is typical."""
    return y * (1 - eps) + 0.5 * eps

def train_multiclass_model(model, dataset, save_interval=None, n_epochs=80, batch_size=1024, lr=1e-3):
    """Train a multi-class PyTorch model with CrossEntropyLoss and optional class weights"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # For multi-class classification, use CrossEntropyLoss
    # If you have class imbalance, you can pass class weights here
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    saved_models = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)  # Shape: (batch_size, num_classes)
            
            # For multi-class, batch_y should be class indices (not one-hot)
            # If batch_y is one-hot, convert it: batch_y = batch_y.argmax(dim=1)
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                batch_y = batch_y.argmax(dim=1)
            
            loss = criterion(logits, batch_y)  # CrossEntropyLoss expects class indices
            loss = (loss * batch_w).mean()     # Apply sample weights
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_X.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_w in val_loader:
                logits = model(batch_X)
                
                # Convert one-hot to class indices if needed
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    batch_y = batch_y.argmax(dim=1)
                
                loss = criterion(logits, batch_y)
                loss = (loss * batch_w).mean()
                total_val_loss += loss.item() * batch_X.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
        
        avg_train_loss = total_train_loss / len(train_dataset)
        avg_val_loss   = total_val_loss / len(val_dataset)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                        f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if save_interval and epoch % save_interval == 0:
            saved_models.append(model.state_dict())

    return model, saved_models

def train_and_reweight_multiclass(nn_arch, data_df_0, mc_df, data_df_1, features, save_interval=None, n_epochs=80, batch_size=1024, lr=1e-3):
    """
    Train a NN classifier for multi-class classification (Data vs MC1 vs MC2), 
    then compute reweighted event weights.
    
    
    Return:
        dict with:
            "model"       : trained NN model
            "w_reco_qcd"  : np.array of reweighted event weights for Data in region 0 to match Data in region 1
    """
    # 0) Prepare inputs for 3-class problem
    p0 = data_df_0[features].to_numpy().astype(np.float32)  
    p1 = mc_df[features].to_numpy().astype(np.float32)    # MC type 1
    p2 = data_df_1[features].to_numpy().astype(np.float32)    # Data type 2

    X = np.vstack([p0, p1, p2])
    y = np.hstack([
        np.zeros(len(p0), dtype=np.int64),  
        np.ones(len(p1), dtype=np.int64),     # MC  = class 1
        np.full(len(p2), 2, dtype=np.int64)  # Data type 2 = class 2
    ])
    
    w = np.hstack([
        np.ones(len(p0), dtype=np.float32),                       
        mc_df['weight'].to_numpy().astype(np.float32),               # MC1 weights
        np.ones(len(p2), dtype=np.float32)                            # Data type 2 weights
    ])
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X, y, w = X[perm], y[perm], w[perm]
    
    # Torch tensors - NOTE: key changes here
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)        # Must be long for CrossEntropy, NO unsqueeze
    w_tensor = torch.tensor(w, dtype=torch.float32)     # NO unsqueeze for weights either
    
    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    
    # 1) Train the model - specify num_classes
    num_classes = 3
    model = nn_arch(len(features), num_classes=num_classes)
    model, saved_models = train_multiclass_model(model, dataset, save_interval=save_interval, n_epochs=n_epochs, batch_size=batch_size, lr=lr)
    # 2) Prediction on DATA only
    results_dict = infer_multiclass(model, data_df_0, features)
    results_dict["model"] = model
    results_dict["saved_models"] = saved_models

    return results_dict

def train_model(model, dataset, n_epochs=80, batch_size=1024, lr=1e-3):
    """Train a given PyTorch model with BCEWithLogitsLoss and weights"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            batch_y_smooth = smooth_labels(batch_y)
            loss = criterion(logits, batch_y_smooth)
            loss = (loss * batch_w).mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_X.size(0)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y, batch_w in val_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss = (loss * batch_w).mean()
                total_val_loss += loss.item() * batch_X.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_val_loss   = total_val_loss / len(val_dataset)
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model

def train_and_reweight(nn_arch, data_df, mc_df, features, n_epochs=80, batch_size=1024, lr=1e-3):
    """
    Train a NN classifier to separate Data vs MC, 
    then compute reweighted event weights for Reco p3 and Reco p2.
    """

    # 0) Prepare inputs
    p1 = data_df[features].to_numpy().astype(np.float32)   # Data
    p2 = mc_df[features].to_numpy().astype(np.float32)     # MC

    X = np.vstack([p1, p2])
    y = np.hstack([
        np.zeros(len(p1), dtype=np.float32),   # Data = 0
        np.ones(len(p2),  dtype=np.float32)    # MC   = 1
    ])
    w = np.hstack([
        np.ones(len(p1), dtype=np.float32),
        mc_df['weight'].to_numpy().astype(np.float32)
    ])

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y, w = X[perm], y[perm], w[perm]

    # Torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    w_tensor = torch.tensor(w, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)

    # 1) Train the model
    model = nn_arch(len(features))
    model = train_model(model, dataset, n_epochs=n_epochs, batch_size=batch_size, lr=lr)

    # 2) Prediction on DATA only
    results_dict = infer_data_to_mc(model, data_df, mc_df, features)
    results_dict["model"] = model
    
    return results_dict
    