#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


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


# In[4]:


import data_preprocess as dpf

xgb_train_norm = dpf.normalize_all(xgb_train)
xgb_test_norm = dpf.normalize_all(xgb_test)
xgb_val_norm = dpf.normalize_all(xgb_val)

tabnet_train_norm = dpf.normalize_all(tabnet_train)
tabnet_test_norm = dpf.normalize_all(tabnet_test)
tabnet_val_norm = dpf.normalize_all(tabnet_val)


# In[5]:


train_indices = tabnet_train_norm.index
valid_indices = tabnet_val_norm.index
test_indices = tabnet_test_norm.index


# In[6]:


target = 'Power (kW)'
features = [ col for col in tabnet_train_norm.columns if col not in target] 

tabnet_X_train = tabnet_train_norm[features].values[train_indices]
tabnet_y_train = tabnet_train_norm[target].values[train_indices]

tabnet_X_valid = tabnet_val_norm[features].values[valid_indices]
tabnet_y_valid = tabnet_val_norm[target].values[valid_indices]

tabnet_X_test = tabnet_test_norm[features].values[test_indices]
tabnet_y_test = tabnet_test_norm[target].values[test_indices]


# In[7]:


target = 'Power (kW)'
features = [ col for col in tabnet_train_norm.columns if col not in target] 


# In[8]:


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


# In[9]:


xgb_X_train = torch_tensor_creator(xgb_train_norm[features])
xgb_y_train = torch_tensor_creator(xgb_train_norm[target])
xgb_X_valid = torch_tensor_creator(xgb_val_norm[features])
xgb_y_valid = torch_tensor_creator(xgb_val_norm[target])
xgb_X_test = torch_tensor_creator(xgb_test_norm[features])
xgb_y_test = torch_tensor_creator(xgb_test_norm[target])


# In[10]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence


# In[11]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")


# In[12]:


# Custom Dataset for dynamic rolling windows
class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, window_size):
        self.x_data = x_data
        self.y_data = y_data
        self.window_size = window_size
        self.indices = []
        for idx in range(len(x_data) - window_size - 217):
            x_window = x_data[idx : idx + self.window_size]
            if not torch.any(torch.isnan(x_window)):
                self.indices.append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        x_window = self.x_data[idx : idx + self.window_size]
        y_window = self.y_data[idx + self.window_size : idx + self.window_size + 218]
        return x_window, y_window



# In[13]:


class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs_packed):
        # Calculate attention scores
        attn_scores = self.attn(inputs_packed.data)
        attn_scores = attn_scores.squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)

        # Calculate attended output
        attended_output = (inputs_packed.data * attn_weights).sum(dim=1)
        attended_output_packed = pack_padded_sequence(attended_output, inputs_packed.batch_sizes, batch_first=True)

        return attended_output_packed


# In[14]:


# Previous code...
class StackedLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StackedLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.lstm_layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))

        # Attention module
        self.attention = AttentionModule(hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs_packed):
        x = inputs_packed
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        # Calculate attention scores and weights
        attention_packed = self.attention(x)

        # Unpack packed sequence and calculate attention
        attended_output_unpacked, _ = pad_packed_sequence(attention_packed, batch_first=True)
        output = self.fc(attended_output_unpacked)

        return output

# Previous code...


# In[15]:


x_train = xgb_X_train
y_train = xgb_y_train 
x_valid = xgb_X_valid
y_valid = xgb_y_valid


# In[23]:


window_size = 2016
# Create DataLoader for training and validation sets with batch_size=1 (Dynamic window approach)
train_dataset = TimeSeriesDataset(x_train, y_train, window_size)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

valid_dataset = TimeSeriesDataset(x_valid, y_valid, window_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


# In[27]:


# Constants
input_size = 64 # 64 features + 2 sin and cos fields
hidden_size = 128
output_size = 218
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_layers = 2  # Number of stacked LSTM layers


# In[28]:


model = StackedLSTMWithAttention(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[31]:


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_dataloader:
        # Move the data to the device
        inputs = inputs.to(device)
        targets = [target.to(device) for target in targets]

        optimizer.zero_grad()
        predictions_packed = model(inputs.float())

        # Unpack sequences and compute the loss
        predictions_unpacked, _ = pad_packed_sequence(predictions_packed, batch_first=True)
        targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True)
        loss = criterion(predictions_unpacked, targets_padded.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")


