from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from os.path import join as pjoin
from src.learning.reweight_base import ReweighterBase
from hep_ml.reweight import GBReweighter
import pickle

class SingleXGBReweighter(ReweighterBase):
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)
        self.metric = 'logloss'

    def prep_data(self, drop_kwd, drop_neg_wgts=True):
        X, y, weights = self.prep_ori_tar(self.ori_data, self.tar_data, drop_kwd, self.weight_column, drop_neg_wgts)
        print("X columns: ")
        print(X.columns)

        X_train, X_test, self.y_train, self.y_test, self.w_train, self.w_test = train_test_split(X, y, weights, test_size=0.3, random_state=42)
        self.dtrain = xgb.DMatrix(X_train, label=self.y_train, weight=self.w_train)
        self.dtest = xgb.DMatrix(X_test, label=self.y_test, weight=self.w_test)
    
    def __define_params(self, max_depth, booster) -> dict:
        self.__params = {"objective": "binary:logistic", "eta": 0.05, "eval_metric": self.metric, "nthread": 4,
                            "subsample": 0.7, "colsample_bytree": 0.8, "seed": 42, "booster": booster, "max_depth": max_depth}
        return self.__params

    def boostingSearch(self, max_depth, num_round, seed=42, booster='gbtree') -> None:
        params = self.__define_params(max_depth, booster)
        metric = self.metric
        
        cv_results = xgb.cv(
            params=params,
            dtrain=self.dtrain,
            metrics=[metric, 'auc', 'rmse'],
            num_boost_round=num_round,
            nfold=6,
            early_stopping_rounds=20,
            as_pandas=True,
            verbose_eval=20
        )
        
        best_round = cv_results[f'test-{metric}-mean'].idxmin()
        print(f"Optimal number of boosting rounds: {best_round}")
        
        self.__cv_results = cv_results
        self.__boost_round = best_round
    
    def train(self, save=False, savename='SingleXGBmodel.json'):
        watchlist = [(self.dtrain, 'train'), (self.dtest, 'test')]
        model = xgb.train(params=self.__params, dtrain=self.dtrain, num_boost_round=self.__boost_round, evals=watchlist, verbose_eval=40)
        if save:
            model.save_model(pjoin(self.results_dir, savename))
        self._model = model
        self._modelname = savename.split('.')[0]
    
    def evaluate(self, save=True):
        y_pred = self._model.predict(self.dtest)
        self.plot_roc(y_pred, self.y_test, self.w_test, save, save_path=pjoin(self.results_dir, f'{self._modelname}_roc.png'))
        
        ax = xgb.plot_importance(self._model, max_num_features=10)
        ax.figure.tight_layout()
        if save:
            ax.figure.savefig(pjoin(self.results_dir, f'{self._modelname}_feature_importance.png'))

        y_pred_binary = (y_pred > 0.5).astype(int)
        print('Accuracy: ', accuracy_score(self.y_test, y_pred_binary))
    
    def load_model(self, model_path):
        self._model = xgb.Booster()
        self._model.load_model(model_path)
        self._modelname = model_path.split('.')[0]
    
    def reweight(self, original, normalize, drop_kwd, save_results=False, save_name='') -> 'pd.DataFrame':
        X, weights, neg_df, dropped_X = self.clean_data(original, drop_kwd, self.weight_column, drop_neg_wgts=True)

        data = xgb.DMatrix(X, weight=weights)
        y_pred = self._model.predict(data)
        
        new_weights = weights * y_pred / (1 - y_pred)
        normalize -= neg_df[self.weight_column].sum()
        new_weights = new_weights * normalize / new_weights.sum()
        
        X = pd.concat([X, dropped_X], axis=1)
        X[self.weight_column] = new_weights
        reweighted = pd.concat([X, neg_df], axis=0)

        if save_results:
            reweighted.to_csv(pjoin(self.results_dir, save_name))
        
        return reweighted
    
class MultiClassXGBReweighter(SingleXGBReweighter):
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)
        self.encoder = LabelEncoder()
        self.metric = 'mlogloss'

    def prep_data(self, drop_kwd, label_col, drop_neg_wgts=True):
        """Preprocess the data into xgboost matrices for multi-class classification."""
        X_ori, w_ori, _, dropped_ori = ReweighterBase.clean_data(self.ori_data, drop_kwd, self.weight_column, drop_neg_wgts=drop_neg_wgts)
        X_tar, w_tar, _, dropped_tar = ReweighterBase.clean_data(self.tar_data, drop_kwd, self.weight_column, drop_neg_wgts=drop_neg_wgts)

        X = pd.concat([X_ori, X_tar], axis=0, ignore_index=True)
        dropped = pd.concat([dropped_ori, dropped_tar], axis=0, ignore_index=True)
        w = pd.concat([w_ori, w_tar], axis=0, ignore_index=True)

        y = self.encoder.fit_transform(dropped[label_col])

        X_train, X_test, self.y_train, self.y_test, self.w_train, self.w_test = train_test_split(X, y, w, test_size=0.3, random_state=42)
        self.dtrain = xgb.DMatrix(X_train, label=self.y_train, weight=self.w_train)
        self.dtest = xgb.DMatrix(X_test, label=self.y_test, weight=self.w_test)

    def __define_params(self, max_depth, booster):
        params = {
            "objective": "multi:softprob",
            "num_class": len(self.encoder.classes_),
            "eta": 0.05,
            "eval_metric": 'mlogloss',
            "nthread": 4,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "seed": 42,
            "booster": booster,
            "max_depth": max_depth
        }
        self.__params = params
        return params

    def evaluate(self, save=True):
        """Evaluate the multi-class XGBoost model using various metrics.
        
        Parameters:
        - save: Boolean indicating whether to save the evaluation plots
        """
        y_pred_proba = self._model.predict(self.dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Overall Accuracy: {accuracy:.4f}')
        
        # Plot feature importance
        ax = xgb.plot_importance(self._model, max_num_features=10)
        ax.figure.tight_layout()
        if save:
            ax.figure.savefig(pjoin(self.results_dir, f'{self._modelname}_feature_importance.png'))
        
        self.plot_roc(
            pred=y_pred_proba,
            label=self.y_test,
            weight=self.w_test,
            save=save,
            save_path=pjoin(self.results_dir, f'{self._modelname}_roc_curves.png'),
            title='Multi-class ROC Curves'
        )
        
        self.plot_confusion(y_pred, self.y_test, self.w_test, self.encoder.classes_, save, pjoin(self.results_dir, f'{self._modelname}_confusion_matrix.png'))
        self.print_classfication(y_pred, self.y_test, self.w_test)
    
    def print_classfication(self, y_pred, y_test, sample_wgt):
        """Print the classification report for multi-class classification."""
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.encoder.classes_, sample_weight=sample_wgt))
    
    def reweight(self, original, normalize, drop_kwd, save_results=False, save_name='') -> 'pd.DataFrame':
        """Reweight the original dataset using multi-class predictions.
        
        Parameters:
        - original: Original DataFrame to be reweighted
        - normalize: Target normalization (sum of weights)
        - drop_kwd: Keywords for columns to be dropped during preprocessing
        - save_results: Whether to save the reweighted DataFrame
        - save_name: Filename for saving results
        
        Returns:
        - pd.DataFrame: Reweighted DataFrame
        """
        X, weights, neg_df, dropped_X = ReweighterBase.clean_data(original, drop_kwd, self.weight_column, drop_neg_wgts=True)
        
        # Get probability predictions for each class
        data = xgb.DMatrix(X, weight=weights)
        class_probabilities = self._model.predict(data)
        
        # Calculate new weights based on the predicted probabilities
        # Using the sum of (p_target / p_original) for each class
        n_classes = len(self.encoder.classes_)
        weight_ratios = np.zeros(len(X))
        
        for i in range(n_classes):
            # Add small epsilon to avoid division by zero
            class_prob = np.clip(class_probabilities[:, i], 1e-7, 1-1e-7)
            weight_ratios += class_prob / (1 - class_prob)
        
        new_weights = weights * weight_ratios / n_classes
        
        # Normalize the weights
        normalize -= neg_df[self.weight_column].sum()
        new_weights = new_weights * normalize / new_weights.sum()
        
        # Reconstruct the DataFrame with new weights
        X = pd.concat([X, dropped_X], axis=1)
        X[self.weight_column] = new_weights
        reweighted = pd.concat([X, neg_df], axis=0)
        
        if save_results:
            reweighted.to_csv(pjoin(self.results_dir, save_name))
        
        return reweighted

class MultipleXGBReweighter(SingleXGBReweighter):
    def prep_data(self, drop_kwd, drop_neg_wgts=True):
        """Preprocess the data into xgboost matrices.
        Drop columns containing the keywords in `drop_kwd` and negatively weighted events if `drop_neg_wgts` is True."""
        X_target, wgt_tar, _, _ = self.clean_data(self.tar_data, drop_kwd, self.weight_column, drop_neg_wgts=drop_neg_wgts)
        X_original, wgt_ori, _, _ = self.clean_data(self.ori_data, drop_kwd, self.weight_column, drop_neg_wgts=drop_neg_wgts)

        self.X_tar_train, self.X_tar_test, self.wgt_tar_train, self.wgt_tar_test = train_test_split(X_target, wgt_tar, test_size=0.3, random_state=42)
        self.X_ori_train, self.X_ori_test, self.wgt_ori_train, self.wgt_ori_test = train_test_split(X_original, wgt_ori, test_size=0.3, random_state=42)

    def train(self, n_estimator, lr, max_depth, min_samples_leaf, save=True, savename="MultiXGBmodel.pkl", gb_args={'subsample': 0.4, 'max_features': 0.75}):
        """Train the reweighting model.

        Parameters:
        - `n_estimator`: Number of boosting rounds (50-100)
        - `lr`: Learning rate
        - `max_depth`: Maximum depth of the trees"""
        model = GBReweighter(n_estimators=n_estimator, learning_rate=lr, max_depth=max_depth, min_samples_leaf=min_samples_leaf, gb_args=gb_args)
        model.fit(self.X_ori_train, self.X_tar_train, self.wgt_ori_train, self.wgt_tar_train)
        self._model = model

        if save:
            with open(pjoin(self.results_dir, savename), 'wb') as f:
                pickle.dump(model, f)

        self._modelname = savename.split('.')[0]

    def reweight(self, original, normalize, drop_kwd, save_results=False, save_name=''):
        X, weights, neg_df, dropped_X = self.clean_data(original, drop_kwd, self.weight_column, drop_neg_wgts=True)

        new_weights = self._model.predict_weights(X, weights)
        normalize = normalize - neg_df[self.weight_column].sum()
        new_weights = new_weights * normalize / new_weights.sum()

        X = pd.concat([X, dropped_X], axis=1)
        X[self.weight_column] = new_weights
        reweighted = pd.concat([X, neg_df], axis=0)

        if save_results:
            reweighted.to_csv(pjoin(self.results_dir, save_name))

        return reweighted