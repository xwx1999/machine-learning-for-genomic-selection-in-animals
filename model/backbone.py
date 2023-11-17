import numpy as np
import pandas as pd
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from tqdm import tqdm
import torch.nn as nn
import os
import math

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerRegressor, self).__init__()

        # 嵌入层
        self.q = nn.Linear(input_size, input_size)
        self.k = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU(0.5)
        self.drop = nn.Dropout(0.1)

        # Layer Normalization层
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)



    def forward(self, x):
        x_ = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attention = F.softmax(q @ k.transpose(-2, -1) / (q.size(-1)) ** 0.5, dim=-1)
        x = (attention @ v).squeeze(2)
        x += x_

        # Layer Normalization
        x = self.layer_norm1(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        # Layer Normalization
        x = self.layer_norm2(x)

        x = self.fc2(x)
        x = self.drop(x)
        output = x.squeeze(-1)

        return output

class CNNGWP(nn.Module):
    def __init__(self, filter, kernel, lambda_, data_type):
        super(CNNGWP, self).__init__()
        self.conv1d1 = nn.Conv1d(1, filter, kernel_size=kernel, stride=3)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.conv1d2 = nn.Conv1d(filter, filter, kernel_size=kernel, stride=1)
        self.flatten = nn.Flatten()
        if data_type == 1:
            self.dense = nn.Linear(438950, 1)
        elif data_type == 2:
            self.dense = nn.Linear(392250, 1)
        elif data_type == 4:
            self.dense = nn.Linear(79600, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

        self.lambda_reg = lambda_

    def forward(self, x):
        x = self.conv1d1(x)
        x = self.maxpool1d(x)
        x = self.conv1d2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# Initialize the parameters with He random values
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # Initialize weights with He random values
        nn.init.zeros_(m.bias)  # Initialize biases with zeros

def gblup(X, Y, lambda_reg):
    # 转换为矩阵
    X = np.array(X)
    Y = np.array(Y)

    n = X.shape[0]  # 样本数量
    p = X.shape[1]  # 特征数量

    # 计算相关矩阵
    H = np.dot(X, X.T)  # 基因型矩阵的内积
    H += np.identity(n) * lambda_reg * np.trace(H)  # 添加正则化项

    # 计算权重向量
    w = np.linalg.pinv(H) @ X @ Y

    # # 预测值
    # y_pred = X.T @ w

    return w

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        output = self.fc(lstm_out.view(len(input), -1))
        return output


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
