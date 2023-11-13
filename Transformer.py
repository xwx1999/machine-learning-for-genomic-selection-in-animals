import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from tqdm import tqdm
import torch.nn as nn
import os
import math
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
X = pd.read_csv('Dataset2/genotype.csv',header=None,low_memory=False)
Y = pd.read_csv('Dataset2/mortality_EBV.csv',header=None,low_memory=False)
X = X.iloc[1:,2:]
Y = Y.iloc[2:,1]


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X_item = self.X[index]
        y_item = self.y[index]
        return X_item, y_item

# Split train-test data
Xtrain = X.iloc[:1000].values  # Xtrain
Ytrain = Y.iloc[:1000].values  # ytrain
Xtest = X.iloc[1000:].values  # Xtest
Ytest = Y.iloc[1000:].values  # ytest

Xtrain = Xtrain.astype(np.float32)
Ytrain = Ytrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Ytest = Ytest.astype(np.float32)

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerRegressor, self).__init__()

        # 嵌入层
        self.q = nn.Linear(input_size,input_size)
        self.k = nn.Linear(input_size,input_size)
        self.v = nn.Linear(input_size,input_size)
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x_ = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attention = (q @ k.transpose(-2,-1))
        attention = attention.softmax(dim=-1)
        x = (attention @ v).squeeze(2)
        x += x_
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        output = x.squeeze(0)

        return output


# Initialize the parameters
input_size = len(Xtrain[0])
hidden_size = 128
model = TransformerRegressor(input_size, hidden_size)
model.train()
model = DataParallel(model)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer,100,0.3)


# Convert input to PyTorch format and create datasets
train_dataset = CustomDataset(Xtrain, Ytrain)
test_dataset = CustomDataset(Xtest, Ytest)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_log = []
test_log = []

# Train the model
num_epochs = 300

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    scheduler.step()
    time = "%s"%datetime.now()
    log = [epoch,time,train_loss]
    train_log.append(log)

    # Evaluate the model
    model.eval()
    mse = 0
    # with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        yhat = model(batch_X.unsqueeze(1)).to(device)
        mse = criterion(yhat, batch_y.unsqueeze(1)).item()
    print(str(epoch+1)+":"+str(mse))
    # if mse<73:
    #     break

# Save the model weights
os.makedirs("Dataset4/Transformer",exist_ok=True)
torch.save(model.state_dict(), "Dataset2/Transformer/model_weights_ebv.pth")


# Print the MSE score
time = "%s"%datetime.now()
test_log.append([time, mse])
print("MSE score:", mse)

# Save the results
train_log = pd.DataFrame(train_log, columns=['epoch','time','train loss'])
test_log = pd.DataFrame(test_log, columns=['time','MSE score'])

train_log.to_csv("Dataset2/Transformer/train_log.csv",index=False)
test_log.to_csv("Dataset2/Transformer/test_log.csv",index=False)
