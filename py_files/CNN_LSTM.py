#!/usr/bin/env python
# coding: utf-8

# In[469]:


import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta 
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm
from fastprogress import master_bar, progress_bar
from itertools import cycle
import datetime as dt

# matplotlib 설정
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


# In[470]:


import os
import torch

# Set CUDA_LAUNCH_BLOCKING environment variable to 1
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Now you can import PyTorch and use the CUDA backend
import torch.cuda

# Rest of your PyTorch code here


# In[471]:


xgb_weights = [3.78227234e-02, 5.35090003e-05, 1.02894781e-04, 1.49786865e-04,
 1.04764847e-04, 1.63700344e-04, 7.86159754e-01, 1.66692644e-01,
 1.62399039e-04, 1.36344403e-04, 1.23864607e-04, 3.25831398e-03,
 1.11413574e-04, 1.12267953e-04, 1.12513058e-04, 1.15435665e-04,
 9.74491413e-05, 7.48724633e-05, 7.42438278e-05, 8.17979526e-05,
 9.10188464e-05, 8.10626516e-05, 1.62408571e-04, 9.22517429e-05,
 6.92411995e-05, 6.06398789e-05, 6.15876997e-05, 2.05707460e-04,
 9.76096126e-05, 9.17855941e-05, 7.44235294e-05, 8.87563365e-05,
 7.44791614e-05, 6.22049338e-05, 6.85490304e-05, 6.57211494e-05,
 6.70680165e-05, 1.42811637e-04, 9.89059918e-05, 9.52478367e-05,
 1.08733664e-04, 8.95528574e-05, 1.07477274e-04, 1.15086681e-04,
 8.60139335e-05, 7.83390569e-05, 6.42610321e-05, 1.03312457e-04,
 9.24527631e-05, 8.45081231e-05, 7.72333588e-05, 8.96328202e-05,
 2.81332632e-05, 2.14036554e-04, 3.18860257e-05, 3.27980561e-05,
 5.04529162e-05, 2.01942370e-04, 7.67721576e-05, 5.93836303e-05,
 6.03012268e-05, 2.65900104e-04, 1.67839185e-04, 8.06350436e-05,
 7.12153196e-05]

tabnet_weights = [4.26892227e-06, 0.00000000e+00, 0.00000000e+00, 3.00723057e-02,
0.00000000e+00, 1.81987064e-06, 1.37099386e-01, 3.07597264e-01,
1.17412334e-05, 0.00000000e+00, 0.00000000e+00, 7.68810592e-02,
6.31930090e-04, 8.53033960e-03, 1.69922775e-02, 8.09343194e-03,
1.85524606e-08, 0.00000000e+00, 5.99931009e-03, 0.00000000e+00,
0.00000000e+00, 0.00000000e+00, 7.60999255e-05, 1.14790590e-06,
1.17366399e-01, 1.14198997e-02, 0.00000000e+00, 3.89959469e-06,
0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.32325158e-05,
0.00000000e+00, 6.57038847e-05, 1.35262607e-02, 2.21752146e-05,
0.00000000e+00, 6.23984265e-02, 8.66896181e-03, 0.00000000e+00,
2.79329075e-03, 1.93734536e-06, 9.73141818e-04, 0.00000000e+00,
2.06456979e-06, 5.23236864e-02, 1.94160263e-02, 4.95664089e-03,
8.15198301e-04, 4.23687933e-02, 4.42582323e-03, 1.96317046e-02,
0.00000000e+00, 2.24109549e-03, 0.00000000e+00, 3.52501762e-03,
4.90427532e-03, 0.00000000e+00, 2.95825462e-03, 0.00000000e+00,
4.75344631e-03, 5.37095814e-03, 0.00000000e+00, 8.05441066e-04,
2.21958440e-02]


# In[472]:


train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')
val = pd.read_pickle('val.pkl')

# Create a dictionary to map feature names to tabnet_weights
tabnet_weight_dict_train = {col: weight for col, weight in zip(train.columns, tabnet_weights)}
tabnet_weight_dict_test = {col: weight for col, weight in zip(test.columns, tabnet_weights)}
tabnet_weight_dict_val = {col: weight for col, weight in zip(val.columns, tabnet_weights)}

# Apply tabnet_weights to train, test, and val DataFrames
tabnet_train = train * pd.Series(tabnet_weight_dict_train)
tabnet_test = test * pd.Series(tabnet_weight_dict_test)
tabnet_val = val * pd.Series(tabnet_weight_dict_val)

#Now to do same for xgb
# Create a dictionary to map feature names to tabnet_weights
xgb_weight_dict_train = {col: weight for col, weight in zip(train.columns, xgb_weights)}
xgb_weight_dict_test = {col: weight for col, weight in zip(test.columns, xgb_weights)}
xgb_weight_dict_val = {col: weight for col, weight in zip(val.columns, xgb_weights)}

#xgb_weights to train, test, and val DataFrames
xgb_train = train * pd.Series(xgb_weight_dict_train)
xgb_test = test * pd.Series(xgb_weight_dict_test)
xgb_val = val * pd.Series(xgb_weight_dict_val)


# In[473]:


import data_preprocess as dpf

xgb_train_norm = dpf.normalize_all(xgb_train)
xgb_test_norm = dpf.normalize_all(xgb_test)
xgb_val_norm = dpf.normalize_all(xgb_val)

tabnet_train_norm = dpf.normalize_all(tabnet_train)
tabnet_test_norm = dpf.normalize_all(tabnet_test)
tabnet_val_norm = dpf.normalize_all(tabnet_val)


# In[474]:


train_indices = tabnet_train_norm.index
valid_indices = tabnet_val_norm.index
test_indices = tabnet_test_norm.index


# In[475]:


target = 'Power (kW)'
features = [ col for col in tabnet_train_norm.columns if col not in target] 


# In[476]:


tabnet_X_train = tabnet_train_norm[features].values[train_indices]
tabnet_y_train = tabnet_train_norm[target].values[train_indices]

tabnet_X_valid = tabnet_val_norm[features].values[valid_indices]
tabnet_y_valid = tabnet_val_norm[target].values[valid_indices]

tabnet_X_test = tabnet_test_norm[features].values[test_indices]
tabnet_y_test = tabnet_test_norm[target].values[test_indices]


# In[477]:


def torch_tensor_creator(df):
    # Convert DataFrame to a numpy array
    data_array = df.values

    # Convert numpy array to a PyTorch tensor
    tensor_data = torch.tensor(data_array, dtype=torch.float)


    # Assuming 'data' is your PyTorch tensor
    has_nans = torch.isnan(tensor_data).any().item()

    if has_nans:
        # Assuming 'tensor_data' is your PyTorch tensor containing the data
    # Find the indices of columns with NaN values
        nan_columns_indices = torch.any(torch.isnan(tensor_data), dim=0).nonzero().squeeze()

        # Remove the columns with NaN values
        tensor_data_without_nan = torch.cat(
            [tensor_data[:, i].unsqueeze(1) for i in range(tensor_data.size(1)) if i not in nan_columns_indices],
            dim=1
        )
    else:
        tensor_data_without_nan = tensor_data
    # Assuming 'data' is your PyTorch tensor
    has_nans = torch.isnan(tensor_data_without_nan)

    # Count the number of NaN values in each column
    num_nans_per_column = torch.sum(has_nans, dim=0)

    return tensor_data_without_nan


# In[478]:


xgb_X_train = torch_tensor_creator(xgb_train_norm[features])
xgb_y_train = torch_tensor_creator(xgb_train_norm[target])
xgb_X_valid = torch_tensor_creator(xgb_val_norm[features])
xgb_y_valid = torch_tensor_creator(xgb_val_norm[target])
xgb_X_test = torch_tensor_creator(xgb_test_norm[features])
xgb_y_test = torch_tensor_creator(xgb_test_norm[target])


# In[479]:


print(xgb_X_train.size())
print(xgb_X_test.size())
print(xgb_y_train.size())
print(xgb_y_test.size())
print(xgb_X_valid.size())
print(xgb_y_valid.size())


# In[480]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# In[481]:


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


# In[482]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class EncoderDecoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cnn_kernel_size=3, lstm_num_layers=1):
        super(EncoderDecoderModel, self).__init__()

        # CNN layer
        self.cnn = nn.Conv1d(input_size, hidden_size, cnn_kernel_size, padding=cnn_kernel_size // 2)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for CNN input
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reshape back for LSTM input
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x


# In[483]:


# Assuming you already have the data as tensors, convert them to DataLoader
# Modify the DataLoader to generate batches with the correct shape

# Update the input and output sizes according to your data
input_size = 64  # Number of features in your input data
output_size = 1006  # Replace ??? with the number of classes in your classification problem

# ... (Your previous code)

# Reshape the input data to have a 3-dimensional shape (batch_size, seq_length, input_size)
seq_length = 253
overlap = 20  # Adjust the overlap as needed
xgb_X_train_reshaped = []
for i in range(0, xgb_X_train_dense.shape[0] - seq_length + 1, overlap):
    xgb_X_train_reshaped.append(xgb_X_train_dense[i:i+seq_length])
xgb_X_train_reshaped = torch.stack(xgb_X_train_reshaped)

# Similarly, reshape the training target data
xgb_y_train_reshaped = []
for i in range(0, xgb_y_train.shape[0] - seq_length + 1, overlap):
    xgb_y_train_reshaped.append(xgb_y_train[i:i+seq_length])
xgb_y_train_reshaped = torch.stack(xgb_y_train_reshaped)

# Do the same for the validation data
xgb_X_valid_reshaped = []
for i in range(0, xgb_X_valid_dense.shape[0] - seq_length + 1, overlap):
    xgb_X_valid_reshaped.append(xgb_X_valid_dense[i:i+seq_length])
xgb_X_valid_reshaped = torch.stack(xgb_X_valid_reshaped)

# Similarly, reshape the validation target data
xgb_y_valid_reshaped = []
for i in range(0, xgb_y_valid.shape[0] - seq_length + 1, overlap):
    xgb_y_valid_reshaped.append(xgb_y_valid[i:i+seq_length])
xgb_y_valid_reshaped = torch.stack(xgb_y_valid_reshaped)

train_dataset = TensorDataset(xgb_X_train_reshaped, xgb_y_train_reshaped)
valid_dataset = TensorDataset(xgb_X_valid_reshaped, xgb_y_valid_reshaped)

train_loader = DataLoader(train_dataset, batch_size=253, shuffle=False)  # Adjust batch_size as needed
valid_loader = DataLoader(valid_dataset, batch_size=253, shuffle=False)  # Adjust batch_size as needed


# In[484]:


def evaluate_model(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            valid_loss += loss.item()
    
    return valid_loss / len(valid_loader)

def train_model(model, train_loader, valid_loader, epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate validation loss after each epoch
        valid_loss = evaluate_model(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")



# In[485]:


# Set the hyperparameters
input_size = 64  # Update the input size to match the number of features in your input data
hidden_size = 64
output_size = 253
cnn_kernel_size = 3
lstm_num_layers = 1

# Create the model with the corrected input_size
model = EncoderDecoderModel(input_size, hidden_size, output_size, cnn_kernel_size, lstm_num_layers)

# Train the model
train_model(model, train_loader, valid_loader, epochs=100, learning_rate=0.001)

