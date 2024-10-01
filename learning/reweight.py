from sklearn.model_selection import train_test_split, GridSearchCV
from src.plotting.visutil import CSVPlotter

import pandas as pd
from hep_ml import reweight 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

def drop_likes(df: 'pd.DataFrame', drop_kwd: 'list[str]' = []):
    for kwd in drop_kwd:
        df = df.drop(columns=df.filter(like=kwd).columns, inplace=False)
    return df

class Reweighter():
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
        self.reweighter = None
        self.visualizer = CSVPlotter(results_dir)
    
    def preprocess_data(self, drop_kwd: 'list[str]' = []):
        self.ori_data = drop_likes(self.ori_data, drop_kwd)
        self.tar_data = drop_likes(self.tar_data, drop_kwd)
            
        self.ori_weight = self.ori_data[self.weight_column]
        self.tar_weight = self.tar_data[self.weight_column]
        
        self.ori_data.drop(columns=[self.weight_column], inplace=True)
        self.tar_data.drop(columns=[self.weight_column], inplace=True)
    
    def fit_rwgt(self):
        self.reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=5, min_samples_leaf=200, 
                                   gb_args={'subsample': 0.4})
        ori_train, ori_test, wo_train, wo_test = train_test_split(self.ori_data, self.ori_weight, test_size=0.3, random_state=42)
        tar_train, tar_test, wt_train, wt_test = train_test_split(self.tar_data, self.tar_weight, test_size=0.3, random_state=42)

        self.reweighter.fit(ori_train, tar_train, wo_train, wt_train)

        return ori_test, wo_test, tar_test, wt_test
    
    def pred_n_compare(self, ori_test, ori_wgt, tar_test, tar_wgt, attridict, ratio_ylabel, save_name, title=''):
        new_ori_wgt = self.reweighter.predict_weights(ori_test, ori_wgt)
        ori = ori_test.copy()
        ori['weight'] = new_ori_wgt
        tar_test['weight'] = tar_wgt
        
        self.visualizer.plot_shape([ori, tar_test], ['Reweighted', 'Target'], attridict, ratio_ylabel=ratio_ylabel, save_name=f'{save_name}_rwgt', title=title)

        ori['weight'] = ori_wgt
        tar_test['weight'] = tar_wgt
        self.visualizer.plot_shape([ori, tar_test], ['Original', 'Target'], attridict, ratio_ylabel=ratio_ylabel, save_name=f'{save_name}_ori', title=title)



class DataLoader():
    def __init__(self, data:'pd.DataFrame', target_column):
        """
        Parameters:
        `data`: pandas DataFrame containing the data."""
        self.target_column = target_column
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
       
    def preprocess_data(self, drop_kwd: 'list[str]' = [], keep_kwd: 'list[str]' = []):
        self.y = self.data[self.target_column]

        for kwd in keep_kwd:
            self.data = self.data.filter(like=kwd)
        
        for kwd in drop_kwd:
            self.data.drop(columns=self.data.filter(like=kwd).columns, inplace=True)
        
        
        self.X = self.data.drop(columns=[self.target_column])
        self.X = self.scaler.fit_transform(self.X)
        
    def split_data(self, test_size=0.3, random_state=None):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            
    def get_train_data(self, if_torch=True):
        if if_torch: 
            return torch.tensor(self.X_train.to_numpy(), dtype=torch.float32), torch.tensor(self.y_train.to_numpy(), dtype=torch.long)
        else:
            return self.X_train, self.y_train
    
    def get_test_data(self, if_torch=False):
        if if_torch:
            return torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.long)
        else:
            return self.X_test, self.y_test

class SimpleClassifier(nn.Module):
    def __init__(self, hidden_size=128, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = None
    
    def fit(self, X_train, y_train, batch_size=32, epochs=50):
        X_train_tensor = X_train
        y_train_tensor = y_train 

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        print('Training finished!')
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, X_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def evaluate(self, X_test, y_test):
        predicted = self.predict(X_test)
        accuracy = (predicted == y_test).mean()
        print(f'Test Accuracy: {accuracy:.4f}')

def est_bkg_RegA(df, cri1, cri2, weight_column):
    """
    Estimate the background in region A using the ABCD method.

    Parameters:
    - df: pandas DataFrame containing the data.
    - cri1: String that defines the first criteria for splitting the data.
    - cri2: String that defines the second criteria for splitting the data.

    Returns:
    - Estimated background in region A.
    """
    region_A = df.query(cri1 + " and " + cri2)
    region_B = df.query(cri1 + " and not (" + cri2 + ")")
    region_C = df.query("not (" + cri1 + ") and " + cri2)
    region_D = df.query("not (" + cri1 + ") and not (" + cri2 + ")")
    
    N_A = region_A[weight_column].sum()
    N_B = region_B[weight_column].sum()
    N_C = region_C[weight_column].sum()
    N_D = region_D[weight_column].sum()
    
    if N_D == 0:
        raise Warning("No events in region D. Cannot estimate background for region A.")
        return None
    N_A_background = (N_B * N_C) / N_D
    
    return N_A_background

def binaryBDTReweighter(X_train, y_train, X_test):
    class_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.05],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_clf = XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=1, scale_pos_weight=class_weight)
    grid_search = GridSearchCV(xgb_clf, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    labels = xgb_clf.predict(X_test)
    probabilities = xgb_clf.predict_proba(X_test)


def XGBweighter(original_train, target_train, original_test, target_test, original_weight, target_weight, draw_cols):
    """Reference: https://github.com/arogozhnikov/hep_ml/blob/master/notebooks/DemoReweighting.ipynb"""
    reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, 
                                   gb_args={'subsample': 0.4})
    reweighter.fit(original_train, target_train, original_weight, target_weight)

    gb_weights_test = reweighter.predict_weights(original_test)
    draw_distributions(original_test, target_test, gb_weights_test, draw_cols)
    