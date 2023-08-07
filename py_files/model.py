#!/usr/bin/env python
# coding: utf-8

# In[252]:


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


# In[253]:


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


# In[254]:


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


# In[255]:


import data_preprocess as dpf

xgb_train_norm = dpf.normalize_all(xgb_train)
xgb_test_norm = dpf.normalize_all(xgb_test)
xgb_val_norm = dpf.normalize_all(xgb_val)

tabnet_train_norm = dpf.normalize_all(tabnet_train)
tabnet_test_norm = dpf.normalize_all(tabnet_test)
tabnet_val_norm = dpf.normalize_all(tabnet_val)


# In[256]:


train_indices = tabnet_train_norm.index
valid_indices = tabnet_val_norm.index
test_indices = tabnet_test_norm.index


# In[257]:


target = 'Power (kW)'
features = [ col for col in tabnet_train_norm.columns if col not in target] 


# In[258]:


tabnet_X_train = tabnet_train_norm[features].values[train_indices]
tabnet_y_train = tabnet_train_norm[target].values[train_indices]

tabnet_X_valid = tabnet_val_norm[features].values[valid_indices]
tabnet_y_valid = tabnet_val_norm[target].values[valid_indices]

tabnet_X_test = tabnet_test_norm[features].values[test_indices]
tabnet_y_test = tabnet_test_norm[target].values[test_indices]


# In[259]:


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


# In[260]:


xgb_X_train = torch_tensor_creator(xgb_train_norm[features])
xgb_y_train = torch_tensor_creator(xgb_train_norm[target])
xgb_X_valid = torch_tensor_creator(xgb_val_norm[features])
xgb_y_valid = torch_tensor_creator(xgb_val_norm[target])
xgb_X_test = torch_tensor_creator(xgb_test_norm[features])
xgb_y_test = torch_tensor_creator(xgb_test_norm[target])


# In[261]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the custom sampler for the data loader
class CustomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=432, overlap=10):
        self.data_source = data_source
        self.batch_size = batch_size
        self.overlap = overlap

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size - self.overlap):
            yield indices[start_idx : start_idx + self.batch_size]

    def __len__(self):
        return (len(self.data_source) - self.batch_size) // (self.batch_size - self.overlap) + 1

# Create the custom batch sampler
batch_size = 263
overlap = 10
train_custom_sampler = CustomBatchSampler(range(len(xgb_X_train)), batch_size, overlap)
valid_custom_sampler = CustomBatchSampler(range(len(xgb_X_valid)), batch_size, overlap)

# Create the data loaders using the custom sampler
train_dataset = TensorDataset(xgb_X_train, xgb_y_train)
train_loader = DataLoader(train_dataset, batch_sampler=train_custom_sampler)

valid_dataset = TensorDataset(xgb_X_valid, xgb_y_valid)
valid_loader = DataLoader(valid_dataset, batch_sampler=valid_custom_sampler)

# Define the stacked LSTM with self-attention
class StackedLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_size, output_size):
        super(StackedLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_size = attention_size

        # Stacked LSTM layers
        self.lstm_stack = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
            nn.Softmax(dim=1)
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm_stack(x, (h0.detach(), c0.detach()))

        # Attention mechanism
        attention_weights = self.attention(out)
        attention_out = torch.sum(attention_weights * out, dim=1)

        # Output layer
        output = self.fc(attention_out)

        return output

# Hyperparameters
input_size = 64
hidden_size = 128
num_layers = 2
attention_size = 64
output_size = 1

# Initialize the model
model = StackedLSTMWithAttention(input_size, hidden_size, num_layers, attention_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reshape data to (batch_size, sequence_length, input_size)
        data = data.view(-1, 1000, 64)

        # Forward pass
        outputs = model(data)

        # Flatten the predictions and targets for loss calculation
        outputs = outputs.view(-1)
        target = target.view(-1)

        # Compute the loss
        loss = criterion(outputs, target)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print batch loss
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

# Validation loop
model.eval()
with torch.no_grad():
    total_loss = 0
    for data, target in valid_loader:
        # Reshape data to (batch_size, sequence_length, input_size)
        data = data.view(-1, 1000, 64)

        outputs = model(data)
        outputs = outputs.view(-1)
        target = target.view(-1)
        loss = criterion(outputs, target)
        total_loss += loss.item()

    average_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {average_loss}")


# In[ ]:


# ... (Previous code)

# Validation loop
model.eval()
predictions = []  # List to store predictions
with torch.no_grad():
    for data, target in valid_loader:
        # Reshape data to (batch_size, sequence_length, input_size)
        data = data.view(-1, 1000, 64)

        outputs = model(data)
        outputs = outputs.view(-1)

        # Append predictions to the list
        predictions.append(outputs)

# Concatenate predictions from all batches
predictions = torch.cat(predictions, dim=0)

# Ensure the shapes are compatible
if predictions.shape[0] != xgb_y_valid.shape[0]:
    diff = xgb_y_valid.shape[0] - predictions.shape[0]
    xgb_y_valid = xgb_y_valid[:-diff]

# Calculate RMSE
mse = nn.MSELoss()(predictions, xgb_y_valid)
rmse = torch.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse.item()}")


# In[ ]:


print(outputs)

