from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score 
import pandas as pd
import os
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np
from torch import nn, optim
import torch

from src.plotting.visutil import CSVPlotter
pjoin = os.path.join

def drop_likes(df: 'pd.DataFrame', drop_kwd: 'list[str]' = []):
    """Drop columns containing the keywords in `drop_kwd`."""
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
        self.results_dir = results_dir
    
    @staticmethod
    def clean_data(df_original, label, drop_kwd, wgt_col, drop_neg_wgts=True) -> tuple['pd.DataFrame', 'pd.Series', 'pd.Series', 'pd.DataFrame']:
        """Clean the data by dropping columns containing the keywords in `drop_kwd`.

        Parameters:
        - `label`: Label column
        
        Return
        - `X`: Features, pandas DataFrame
        - `y`: Labels, pandas Series
        - `weights`: Weights, pandas Series
        - `neg_df`: DataFrame containing the events with negative weights."""
        df = df_original.copy()
        neg_df = df[df[wgt_col] < 0]
        df = df[df[wgt_col] > 0] if drop_neg_wgts else df

        print("Dropped ", len(df_original) - len(df), " events with negative weights out of ", len(df_original), " events.")

        drop_kwd.append(wgt_col)
        X = drop_likes(df, drop_kwd)

        y = pd.Series([label] * len(df))

        weights = df[wgt_col]
        
        return X, y, weights, neg_df
    
    @staticmethod
    def prep_ori_tar(ori, tar, drop_kwd, wgt_col, drop_neg_wgts=True):
        """Preprocess the original and target data by dropping columns containing the keywords in `drop_kwd`."""
        
        X_ori, y_ori, w_ori, _ = Reweighter.clean_data(ori, 0, drop_kwd, wgt_col, drop_neg_wgts)
        X_tar, y_tar, w_tar, _ = Reweighter.clean_data(tar, 1, drop_kwd, wgt_col, drop_neg_wgts)
        
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

class SingleXGBReweighter(Reweighter):
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)

    def prep_data(self, drop_kwd, drop_neg_wgts=True):
        """Preprocess the data by dropping columns containing the keywords in `drop_kwd`."""
        X, y, weights = self.prep_ori_tar(self.ori_data, self.tar_data, drop_kwd, self.weight_column, drop_neg_wgts)

        X_train, X_test, self.y_train, self.y_test, self.w_train, self.w_test = train_test_split(X, y, weights, test_size=0.3, random_state=42)
        self.dtrain = xgb.DMatrix(X_train, label=self.y_train, weight=self.w_train)
        self.dtest = xgb.DMatrix(X_test, label=self.y_test, weight=self.w_test)

    def boostingSearch(self, max_depth, num_round, seed=42) -> None:
        """Perform grid search to find the best hyperparameters for the XGBoost model.
        
        Parameters:
        `max_depth`: List of integers containing the maximum depth of the trees.
        `num_round`: List of integers containing the number of boosting rounds."""
        
        params = {"objective": "binary:logistic", "max_depth": max_depth, "eta": 0.1, "eval_metric": 'logloss', "nthread": 4, "seed": seed}
        cv_results = xgb.cv(
            params=params,
            dtrain=self.dtrain,
            num_boost_round=num_round,
            nfold=6,  # Number of CV folds
            early_stopping_rounds=20,  # Stop if no improvement after 10 rounds
            as_pandas=True,  # Return results as a pandas DataFrame
            verbose_eval=20  # Print results every 20 rounds
        )
        print(cv_results)
        best_round = cv_results['test-logloss-mean'].idxmin()
        print(f"Optimal number of boosting rounds: {best_round}")
        
        self.__cv_results = cv_results
        self.__boost_round = best_round
        self.__params = params
    
    def train(self, save=False):
        """Train the XGBoost model. 
        
        Parameters:
        - `save`: Boolean indicating whether to save the model."""
        watchlist = [(self.dtrain, 'train'), (self.dtest, 'test')]
        model = xgb.train(params=self.__params, dtrain=self.dtrain, num_boost_round=self.__boost_round, evals=watchlist, verbose_eval=40)
        if save:
            model.save_model(pjoin(self.results_dir, 'SingleXGBmodel.xgb'))
        
        self._model = model
    
    def evaluate(self):
        """Evaluate the model on the test data."""
        y_pred = self._model.predict(self.dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        print('Accuracy: ', accuracy_score(self.y_test, y_pred_binary))

        roc_auc = roc_auc_score(self.y_test, y_pred)
        print('ROC AUC Score: ', roc_auc)
    
    def reweight(self, original, normalize, drop_kwd, save_results=False, save_name=''):
        """Reweight the original data.
        
        Parameters
        - `original`: pandas DataFrame containing the original data to be reweighted
        - `normalize`: constant to which the weights are normalized"""

        #! note a potential bug: X does not contain the dropped columns but neg_df does
        X, y, weights, neg_df = self.clean_data(original, 0, drop_kwd, self.weight_column, drop_neg_wgts=True)

        data = xgb.DMatrix(X, label=y, weight=weights)
        y_pred = self._model.predict(data)
        
        new_weights = weights * y_pred / (1 - y_pred)
        normalize -= neg_df[self.weight_column].sum()
        new_weights = new_weights * normalize / new_weights.sum()
        
        if save_results:
            X[self.weight_column] = new_weights
            reweighted = pd.concat([X, neg_df], ignore_index=True)
            reweighted.to_csv(pjoin(self.results_dir, save_name))
        
        return reweighted
    
    def plot_rwgt(self, original, target, drop_kwd, original_name, target_name, kin_var, range):
        pass

        # self.draw_distributions(X_filtered, target, weights_filtered, target['weight'], 
        #                         f'{original_name} (Before Rwgt)', target_name, 
        #                         kin_var, bins=10, range=range, save_path=pjoin(self.results_dir, f'{kin_var}.png'))
        # self.draw_distributions(X_filtered, target, new_weights, target['weight'], 
        #                         f'{original_name} (After Rwgt)', target_name,
        #                         kin_var, bins=10, range=range, save_path=pjoin(self.results_dir, f'{kin_var}.png'))
    
class MultipleXGBReweighter(Reweighter):
    def preprocess_data(self, drop_kwd: 'list[str]' = [], drop_neg_wgts=True):
        """Preprocess the data by dropping columns containing the keywords in `drop_kwd`."""
        if drop_neg_wgts:
            self.ori_data = self.ori_data[self.ori_data[self.weight_column] > 0]
            self.tar_data = self.tar_data[self.tar_data[self.weight_column] > 0]

        self.ori_weight = self.ori_data[self.weight_column]
        self.tar_weight = self.tar_data[self.weight_column]

        self.ori_data = drop_likes(self.ori_data, drop_kwd)
        self.tar_data = drop_likes(self.tar_data, drop_kwd)
            
        self.ori_data.drop(columns=[self.weight_column], inplace=True)
        self.tar_data.drop(columns=[self.weight_column], inplace=True)
    
    def gridSearch(self, param_grid, scoring='roc_auc', cv=5, n_jobs=-1):
        estimator = reweight.GBReweighter()
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    
    def fit_rwgt(self, **kwargs) -> 'reweight.GBReweighter':
        """Fit the reweighter on the original and target data."""
        self.reweighter = reweight.GBReweighter(**kwargs)
        ori_train, ori_test, wo_train, wo_test = train_test_split(self.ori_data, self.ori_weight)
        tar_train, tar_test, wt_train, wt_test = train_test_split(self.tar_data, self.tar_weight)

        self.reweighter.fit(ori_train, tar_train, wo_train, wt_train)

        self.o_train = ori_train
        self.o_test = ori_test
        self.wo_train = wo_train
        self.wo_test = wo_test

        self.t_train = tar_train
        self.t_test = tar_test
        self.wt_train = wt_train
        self.wt_test = wt_test

        return self.reweighter
    
    def add_test_data(self, ori_test, tar_test, ori_wgt, tar_wgt):
        self.o_test = pd.concat([self.o_test, ori_test], ignore_index=True)
        self.wo_test = pd.concat([self.wo_test, ori_wgt], ignore_index=True)
        self.t_test = pd.concat([self.t_test, tar_test], ignore_index=True)
        self.wt_test = pd.concat([self.wt_test, tar_wgt], ignore_index=True)

    def get_new_weights_train(self):
        """Get the new weights for the original training data."""
        return self.reweighter.predict_weights(self.o_train, self.wo_train)
    
    def get_new_weights_test(self):
        return self.reweighter.predict_weights(self.o_test, self.wo_test)
    
    def pred_n_compare(self, attridict, ratio_ylabel, save_name, title=''):
        new_ori_wgt = self.reweighter.predict_weights(self.o_test, self.wo_test)
        ori = self.o_test.copy()
        ori['weight'] = new_ori_wgt
        self.t_test['weight'] = self.wt_test
        
        self.visualizer.plot_shape([ori, self.t_test], ['Reweighted', 'Target'], attridict, ratio_ylabel=ratio_ylabel, save_name=f'{save_name}_rwgt', title=title)

        ori['weight'] = self.wo_test
        self.visualizer.plot_shape([ori, self.t_test], ['Original', 'Target'], attridict, ratio_ylabel=ratio_ylabel, save_name=f'{save_name}_ori', title=title)
    
    

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
    