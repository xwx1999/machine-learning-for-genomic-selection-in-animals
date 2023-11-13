import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from tqdm import tqdm
import torch.nn as nn
import os
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
X = pd.read_csv('Dataset4/QTLMAS2010ny012.csv',header=None,low_memory=False)
X = X.iloc[:,1:-1]
Y = pd.read_csv('Dataset4/QTLMAS2010ny012.csv',header=None,low_memory=False)
Y = Y.iloc[:,0]


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
Xtrain = X.iloc[:2226].values  # Xtrain
Ytrain = Y.iloc[:2226].values  # ytrain
Xtest = X.iloc[2226:].values  # Xtest
Ytest = Y.iloc[2226:].values  # ytest

Xtrain = Xtrain.astype(np.float32)
Ytrain = Ytrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Ytest = Ytest.astype(np.float32)

class CNNGWP(nn.Module):
    def __init__(self, filter, kernel, lambda_):
        super(CNNGWP, self).__init__()
        self.conv1d1 = nn.Conv1d(1, filter, kernel_size=kernel, stride=3)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.conv1d2 = nn.Conv1d(filter, filter, kernel_size=kernel, stride=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(79600, 1)
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


filter = 50
kernel = 25
lambda_reg = 0.5

# Create the CNNGWP model
model = CNNGWP(filter, kernel, lambda_reg)
model.apply(init_weights)
# 加载模型权重，修改state_dict中的键名称
# model_state_dict = torch.load('Dataset4/CNNGWP/model_weights_ebv_conv2.pth')
# renamed_state_dict = {}
# for k, v in model_state_dict.items():
#     name = k.replace("module.", "")  # 去除键名称中的 "module."
#     renamed_state_dict[name] = v
# model.load_state_dict(renamed_state_dict)
model.train()
model = DataParallel(model).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer,100,0.5)


#test
example = torch.rand((1, 9722))
example = example.unsqueeze(0) # 将example的维度调整为3维
print(model(example).size())


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
    loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()
    scheduler.step()
    t = "%s"%datetime.now()
    log = [epoch,t,loss.item()]
    train_log.append(log)

    # Evaluate the model
    model.eval()

    # with torch.no_grad():
    start_time = time.time()
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        yhat = model(batch_X.unsqueeze(1)).to(device)
        mse = criterion(yhat, batch_y.unsqueeze(1)).item()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行耗时: {elapsed_time:.2f}秒")
    print(str(epoch+1)+":"+str(mse))
    # if mse<73:
    #     break

# Save the model weights
torch.save(model.state_dict(), "Dataset4/CNNGWP/model_weights_ebv_conv2.pth")


# Print the MSE score
t = "%s"%datetime.now()
test_log.append([t, mse])
print("MSE score:", mse)

# Save the results
train_log = pd.DataFrame(train_log, columns=['epoch','time','train loss'])
test_log = pd.DataFrame(test_log, columns=['time','MSE score'])

train_log.to_csv("Dataset4/CNNGWP/train_log_conv2.csv",index=False)
test_log.to_csv("Dataset4/CNNGWP/test_log_conv2.csv",index=False)
