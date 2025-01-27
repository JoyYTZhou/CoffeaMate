from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix


def add_hidden_layer(layers, in_dim, hidden_dims, activation):
    """Add hidden layers to the model."""
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation)
        in_dim = h_dim

class ReweighterBase():
    """Base class for reweighting.

    Attributes:
    - `ori_data`: pandas DataFrame containing the original data.
    - `tar_data`: pandas DataFrame containing the target data.
    - `weight_column`: Column name for the weights.
    - `ori_weight`: Weights for the original data.
    - `tar_weight`: Weights for the target data.
    - `results_dir`: Base directory to save the results, e.g. plots."""
    def __init__(self, ori_data:'pd.DataFrame', tar_data:'pd.DataFrame', weight_column, results_dir):
        """
        Parameters:
        `ori_data`: pandas DataFrame containing the original data.
        `tar_data`: pandas DataFrame containing the target data."""
        self.ori_data = ori_data
        self.tar_data = tar_data
        self.weight_column = weight_column
        self.ori_weight = None
        self.tar_weight = None
        self.results_dir = results_dir

    @staticmethod
    def drop_likes(df: 'pd.DataFrame', drop_kwd: 'list[str]' = []):
        """Drop columns containing the keywords in `drop_kwd`."""
        dropped = pd.DataFrame()
        for kwd in drop_kwd:
            cols_to_drop = df.filter(like=kwd).columns
            dropped = pd.concat([dropped, df[cols_to_drop]], axis=1)
            df = df.drop(columns=df.filter(like=kwd).columns, inplace=False)
        return df, dropped
    
    @staticmethod
    def clean_data(df_original, drop_kwd, wgt_col, drop_wgts=True, drop_neg_wgts=True) -> tuple['pd.DataFrame', 'pd.Series', 'pd.DataFrame', 'pd.DataFrame']:
        """Clean the data by dropping columns containing the keywords in `drop_kwd`.

        Parameters:
        - `drop_kwd`: List of keywords to drop.
        - `wgt_col`: Column name for the weights.
        
        Return
        - `X`: Features, pandas DataFrame
        - `weights`: Weights, pandas Series
        - `neg_df`: DataFrame containing the events with negative weights.
        - `dropped_X`: DataFrame containing the dropped columns.
        """
        df = df_original.copy()
        neg_df = df[df[wgt_col] < 0]
        df = df[df[wgt_col] > 0] if drop_neg_wgts else df

        print("Dropped ", len(df_original) - len(df), " events with negative weights out of ", len(df_original), " events.")

        if drop_wgts: drop_kwd.append(wgt_col)
        X, dropped_X = ReweighterBase.drop_likes(df, drop_kwd)

        weights = df[wgt_col]
        
        return X, weights, neg_df, dropped_X
    
    @staticmethod
    def int_label(label, length):
        """Return the label as a pandas Series"""
        return pd.Series([label]*length)
    
    @staticmethod
    def prep_ori_tar(ori, tar, drop_kwd, wgt_col, drop_neg_wgts=True, drop_wgts=True):
        """Preprocess the original and target data by dropping columns containing the keywords in `drop_kwd`."""
        
        X_ori, w_ori, _, _= ReweighterBase.clean_data(ori, drop_kwd, wgt_col, drop_neg_wgts=drop_neg_wgts, drop_wgts=drop_wgts)
        y_ori = ReweighterBase.int_label(0, len(X_ori))
        X_tar, w_tar, _, _ = ReweighterBase.clean_data(tar, drop_kwd, wgt_col, drop_neg_wgts=drop_neg_wgts, drop_wgts=drop_wgts)
        y_tar = ReweighterBase.int_label(1, len(X_tar))
        
        return pd.concat([X_ori, X_tar], ignore_index=True, axis=0), pd.concat([y_ori, y_tar], ignore_index=True, axis=0), pd.concat([w_ori, w_tar], ignore_index=True, axis=0)
    
    @staticmethod
    def draw_distributions(original, target, o_wgt, t_wgt, original_label, target_label, column, bins=10, range=None, save_path=False):
        """Draw the distributions of the original and target data. Normalized."""
        hist_settings = {'bins': bins, 'density': True, 'alpha': 0.5}
        plt.figure(figsize=[12, 7])
        xlim = np.percentile(np.hstack([target[column]]), [0.01, 99.99])
        range = xlim if range is None else range
        plt.hist(original[column], weights=o_wgt, range=range, label=original_label, **hist_settings)
        plt.hist(target[column], weights=t_wgt, range=range, label=target_label, **hist_settings)
        plt.legend(loc='best')
        plt.title(column)
        if save_path:
            plt.savefig(save_path)
    
    @staticmethod
    def compute_nn_auc(model, data_loader, device, save, save_path, title='ROC Curve'):
        """Compute the AUC score for a nn model."""
        all_labels = []
        all_preds = []
        all_weights = []
        
        model.eval()
        print("Using device: ", device)
        model.to(device)
        with torch.no_grad():
            for data, label, weight in data_loader:
                data, label, weight = data.to(device), label.to(device), weight.to(device)
                #! Bug here
                pred = model(data).squeeze()
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_weights.extend(weight.cpu().numpy())
        
        ReweighterBase.plot_roc(all_preds, all_labels, all_weights, save, save_path, title)
        
    @staticmethod
    def plot_roc(pred, label, weight, save, save_path, title='ROC Curve'):
        """Plot the ROC curve for both binary and multi-class classification.
        
        Parameters:
        - pred: Model predictions. For binary classification, a 1D array of probabilities.
                For multi-class, a 2D array where each column represents class probabilities.
        - label: True labels. For binary classification, a 1D array.
                For multi-class, a 1D array with class indices.
        - weight: Sample weights
        - save: Boolean indicating whether to save the plot
        - save_path: Path where to save the plot
        - title: Title of the plot
        """
        plt.figure(figsize=(10, 8))
        
        pred = np.array(pred)
        label = np.array(label)
        weight = np.array(weight)
        
        if len(pred.shape) == 2:
            n_classes = pred.shape[1]
            for i in range(n_classes):
                binary_label = (label == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_label, pred[:, i], sample_weight=weight)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
                print(f'ROC AUC Score for Class {i}: {auc_score:.3f}')
        
        else:
            fpr, tpr, _ = roc_curve(label, pred, sample_weight=weight)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
            print(f'ROC AUC Score: {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
        
        if save:
            plt.savefig(save_path)
            plt.close()

    @staticmethod
    def plot_confusion(pred, label, weight, classes, save, save_path):
        """Plot the confusion matrix for both binary and multi-class classification.
        
        Parameters:
        - pred: Model predictions. For binary classification, a 1D array of probabilities.
                For multi-class, a 2D array
        - classes: List of class names"""
        cm = confusion_matrix(label, pred, sample_weight=weight)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save:
            plt.savefig(save_path)
        plt.close()
        
class WeightedDataset(Dataset):
    """Dataset class for weighted data."""
    def __init__(self, dataframe, feature_columns, weight_column):
        self.data = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32)
        self.weights = torch.tensor(dataframe[weight_column].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        weight = self.weights[idx]
        return torch.tensor(data, dtype=torch.float), torch.tensor(weight, dtype=torch.float)

class Generator(nn.Module):
    """Generator class for the reweighting model."""
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()

        layers = []
        add_hidden_layer(layers, input_dim, hidden_dims, nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        layers = []
        add_hidden_layer(layers, input_dim, hidden_dims, nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)