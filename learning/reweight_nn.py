from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from os.path import join as pjoin
import pandas as pd
import numpy as np
from src.learning.reweight_base import ReweighterBase, WeightedDataset, Generator, Discriminator

def check_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

def weighted_bce_loss(predictions, targets, weights):
    return torch.mean(weights * nn.BCELoss(reduction='none')(predictions, targets))
 
class SingleNNReweighter(ReweighterBase):
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)
        self.scaler = MinMaxScaler()

    def prep_data(self, drop_kwd, drop_neg_wgts=True):
        """Preprocess the data into TensorDataset objects.
        Drop columns containing the keywords in `drop_kwd` and negatively weighted events if `drop_neg_wgts` is True."""
        X, y, weights = self.prep_ori_tar(self.ori_data, self.tar_data, drop_kwd, self.weight_column, drop_wgts=True, drop_neg_wgts=drop_neg_wgts)
        features = list(X.columns)

        X[features] = self.scaler.fit_transform(X[features])

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X.values, y.values, weights.values, test_size=0.3, random_state=42)

        self.dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(w_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1), torch.tensor(w_val, dtype=torch.float32))
        self.features = features

    def train(self, num_epochs, hidden_dims, batch_size, lr, save=True, savename='SingleNNmodel.pth', save_interval=30):
        """Train the discriminator model.

        Parameters:
        - `num_epochs`: Number of epochs
        - `hidden_dims`: List of integers containing the number of hidden units in each layer.
        - `batch_size`: Batch size
        - `eval`: Boolean indicating whether to evaluate the model on the validation data."""
        device = check_device()
        input_dim = len(self.features)

        model = Discriminator(input_dim, hidden_dims).to(device)

        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            val_running_loss = 0.0

            for batch_data, batch_labels, batch_weights in train_loader:
                batch_data, batch_labels, batch_weights = batch_data.to(device), batch_labels.to(device), batch_weights.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                weighted_loss = (loss * batch_weights).mean()
                weighted_loss.backward()
                optimizer.step()
                running_loss += weighted_loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            model.eval()

            with torch.no_grad():
                for val_data, val_labels, val_weights in val_loader:
                    val_data, val_labels, val_weights = val_data.to(device), val_labels.to(device), val_weights.to(device)
                    val_outputs = model(val_data)
                    val_loss = criterion(val_outputs, val_labels)
                    weighted_val_loss = (val_loss * val_weights).mean()
                    val_running_loss += weighted_val_loss.item()

            val_loss = val_running_loss / len(val_loader)
            val_losses.append(weighted_val_loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss}, Val Loss: {val_loss}")

            if (epoch + 1) % save_interval == 0 and save:
                save_path = f"{savename.split('.')[0]}_{epoch+1}.pth"
                torch.save(model.state_dict(), pjoin(self.results_dir, save_path))
                print(f"Model saved at {save_path}")

        self.history = {'train': train_losses, 'val': val_losses}
        self.model = model

        if save:
            torch.save(model.state_dict(), pjoin(self.results_dir, savename))
            print(f"Final Model saved to {savename}")

        self._name = savename.split('.')[0]

    def load_model(self, model_name, hidden_dim):
        """Load the model from a saved file."""
        model = Discriminator(len(self.features), hidden_dim)
        model.load_state_dict(torch.load(pjoin(self.results_dir, model_name)))
        self.model = model
        self._name = model_name.split('.')[0]

    def evaluate(self):
        """Compute the AUC score for the model."""
        device = check_device()
        self.compute_nn_auc(self.model, DataLoader(self.val_dataset, batch_size=64, shuffle=False), device, True, pjoin(self.results_dir, f'{self._name}_val_roc.png'), 'Validation ROC Curve')
        self.compute_nn_auc(self.model, DataLoader(self.dataset, batch_size=64, shuffle=False), device, True, pjoin(self.results_dir, f'{self._name}_train_roc.png'), 'Training ROC Curve')

    def visualize(self, save=False):
        """Visualize the training and validation losses."""
        plt.figure(figsize=[12, 7])
        plt.plot(self.history['train'], label='Train')
        plt.plot(self.history['val'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(pjoin(self.results_dir, 'train_val_losses.png'))
        else:
            plt.show()

    def reweight(self, original, normalize, drop_kwd, save_results=False, save_name='') -> 'pd.DataFrame':
        """Reweight the original data.

        Parameters
        - `original`: pandas DataFrame containing the original data to be reweighted
        - `normalize`: constant to which the weights are normalized"""

        X_df, weights, neg_df, _ = self.clean_data(original, drop_kwd, self.weight_column, drop_neg_wgts=True)
        X = self.scaler.transform(X_df)
        data = torch.tensor(X, dtype=torch.float32)
        data = DataLoader(data, batch_size=512, shuffle=False)

        device = check_device()
        model = self.model.to(device)
        model.eval()

        new_weights = torch.zeros(len(weights), dtype=torch.float32)
        start_idx = 0

        for batch in data:
            batch_size = batch.size(0)
            with torch.no_grad():
                outputs = model(batch.to(device)).cpu()
                new_weights[start_idx:start_idx+batch_size] = outputs.squeeze()
            start_idx += batch_size

        new_weights = new_weights.numpy()

        epsilon = 1e-6  # Small value to prevent division by zero
        new_weights = np.clip(new_weights, epsilon, 1 - epsilon)

        new_weights = weights * new_weights / (1 - new_weights)
        normalize -= neg_df[self.weight_column].sum()
        new_weights = new_weights * normalize / new_weights.sum()

        X_df[self.weight_column] = new_weights
        reweighted = pd.concat([X_df, neg_df], ignore_index=True)

        if save_results:
            reweighted.to_csv(pjoin(self.results_dir, save_name))

        return reweighted

class GANReweighter(SingleNNReweighter):
    def __init__(self, ori_data, tar_data, weight_column, results_dir):
        super().__init__(ori_data, tar_data, weight_column, results_dir)

    def prep_data(self, drop_kwd, drop_neg_wgts=True) -> tuple[WeightedDataset, pd.DataFrame]:
        """Preprocess the data into WeightedDataset objects.
        Drop columns containing the keywords in `drop_kwd` and negatively weighted events if `drop_neg_wgts` is True."""
        target_df, _,  _, _, _ = self.clean_data(self.tar_data, drop_kwd, self.weight_column, None, False, drop_neg_wgts)
        features = list(target_df.columns)
        features.remove(self.weight_column)
        target_df[features] = self.scaler.fit_transform(target_df[features])
        target_df["weight"] /= target_df["weight"].sum()
        target_dataset = WeightedDataset(target_df, features, self.weight_column)

        noise_df, _, _, _, _ = self.clean_data(self.ori_data, drop_kwd, self.weight_column, None, False, drop_neg_wgts)
        noise_df[features] = self.scaler.transform(noise_df[features])
        noise_df["weight"] /= noise_df["weight"].sum()

        self.target_dataset = target_dataset
        self.noise_df = noise_df
        self.feature_cols = features

        return self.target_dataset, self.noise_df

    def train(self, num_epochs, hidden_dims, batch_size):
        mps_device = torch.device('mps')

        target_loader = DataLoader(self.tar_data, batch_size, shuffle=True)
        noise_data = self.noise_df[self.feature_cols].values()
        noise_weights = self.noise_df[self.weight_column].values()

        input_dim = len(self.feature_cols)
        # generator = Generator(input_dim, input_dim, hidden_dims).to(mps_device)
        discriminator = Discriminator(input_dim, hidden_dims).to(mps_device)
        generator = Discriminator(input_dim, hidden_dims).to(mps_device)

        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)

        G_losses = []
        D_losses = []

        for epoch in range(num_epochs):
            for real_samples, real_weights in target_loader:
                real_samples, real_weights = real_samples.to(mps_device), real_weights.to(mps_device)
                batch_size = real_samples.shape[0]

                optimizer_D.zero_grad()

                real_labels = torch.ones(batch_size, 1).to(mps_device)
                real_predictions = discriminator(real_samples)
                real_loss = weighted_bce_loss(real_predictions, real_labels, real_weights)

                noise_indices = np.random.choice(len(noise_data), size=batch_size, p=noise_weights)
                noise = torch.tensor(noise_data[noise_indices], dtype=torch.float32).to(mps_device)
                fake_weights = generator(noise)
                fake_labels = torch.zeros(batch_size, 1).to(mps_device)
                fake_predictions = discriminator(noise)
                fake_loss = weighted_bce_loss(fake_predictions, fake_labels, fake_weights)

        pass

# class DataLoader():
#     def __init__(self, data:'pd.DataFrame', target_column):
#         """
#         Parameters:
#         `data`: pandas DataFrame containing the data."""
#         self.target_column = target_column
#         self.data = data
#         self.X = None
#         self.y = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.scaler = StandardScaler()

#     def preprocess_data(self, drop_kwd: 'list[str]' = [], keep_kwd: 'list[str]' = []):
#         self.y = self.data[self.target_column]

#         for kwd in keep_kwd:
#             self.data = self.data.filter(like=kwd)

#         for kwd in drop_kwd:
#             self.data.drop(columns=self.data.filter(like=kwd).columns, inplace=True)


#         self.X = self.data.drop(columns=[self.target_column])
#         self.X = self.scaler.fit_transform(self.X)

#     def split_data(self, test_size=0.3, random_state=None):
#         self.X_train, self.X_test, self.y_train, self.y_test = \
#             train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

#     def get_train_data(self, if_torch=True):
#         if if_torch:
#             return torch.tensor(self.X_train.to_numpy(), dtype=torch.float32), torch.tensor(self.y_train.to_numpy(), dtype=torch.long)
#         else:
#             return self.X_train, self.y_train

#     def get_test_data(self, if_torch=False):
#         if if_torch:
#             return torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.long)
#         else:
#             return self.X_test, self.y_test

# class SimpleClassifier(nn.Module):
#     def __init__(self, hidden_size=128, num_classes=2):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.LazyLinear(hidden_size),
#             nn.ReLU(),
#             nn.LazyLinear(num_classes)
#         )
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.train_loader = None

#     def fit(self, X_train, y_train, batch_size=32, epochs=50):
#         X_train_tensor = X_train
#         y_train_tensor = y_train

#         train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#         self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#         for epoch in range(epochs):
#             for inputs, labels in self.train_loader:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()

#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#         print('Training finished!')

#     def forward(self, x):
#         return self.model(x)

#     def predict(self, X_test):
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

#         with torch.no_grad():
#             outputs = self.model(X_test_tensor)
#             _, predicted = torch.max(outputs, 1)

#         return predicted.numpy()

#     def evaluate(self, X_test, y_test):
#         predicted = self.predict(X_test)
#         accuracy = (predicted == y_test).mean()
#         print(f'Test Accuracy: {accuracy:.4f}')

# def est_bkg_RegA(df, cri1, cri2, weight_column):
#     """
#     Estimate the background in region A using the ABCD method.

#     Parameters:
#     - df: pandas DataFrame containing the data.
#     - cri1: String that defines the first criteria for splitting the data.
#     - cri2: String that defines the second criteria for splitting the data.

#     Returns:
#     - Estimated background in region A.
#     """
#     region_A = df.query(cri1 + " and " + cri2)
#     region_B = df.query(cri1 + " and not (" + cri2 + ")")
#     region_C = df.query("not (" + cri1 + ") and " + cri2)
#     region_D = df.query("not (" + cri1 + ") and not (" + cri2 + ")")

#     N_A = region_A[weight_column].sum()
#     N_B = region_B[weight_column].sum()
#     N_C = region_C[weight_column].sum()
#     N_D = region_D[weight_column].sum()

#     if N_D == 0:
#         raise Warning("No events in region D. Cannot estimate background for region A.")
#         return None
#     N_A_background = (N_B * N_C) / N_D

#     return N_A_background

# def binaryBDTReweighter(X_train, y_train, X_test):
#     class_weight = (len(y_train) - y_train.sum()) / y_train.sum()
#     param_grid = {
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.1, 0.01, 0.05],
#         'n_estimators': [50, 100, 200],
#         'subsample': [0.8, 1.0],
#         'colsample_bytree': [0.8, 1.0]
#     }
#     xgb_clf = XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=1, scale_pos_weight=class_weight)
#     grid_search = GridSearchCV(xgb_clf, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
#     grid_search.fit(X_train, y_train)

#     labels = xgb_clf.predict(X_test)
#     probabilities = xgb_clf.predict_proba(X_test)


# def XGBweighter(original_train, target_train, original_test, target_test, original_weight, target_weight, draw_cols):
#     """Reference: https://github.com/arogozhnikov/hep_ml/blob/master/notebooks/DemoReweighting.ipynb"""
#     reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000,
#                                    gb_args={'subsample': 0.4})
#     reweighter.fit(original_train, target_train, original_weight, target_weight)

#     gb_weights_test = reweighter.predict_weights(original_test)
#     draw_distributions(original_test, target_test, gb_weights_test, draw_cols)

