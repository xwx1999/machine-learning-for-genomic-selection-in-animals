import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from tqdm import tqdm
import torch.nn as nn
# import lightgbm as lgb
import argparse
import os
import math
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from model.backbone import TransformerRegressor, CNNGWP, init_weights, gblup, LSTMModel, Autoencoder
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from dataset.SnpSequence import SNPDataset, CustomDataset

def run(args):
    # GPU Settings
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Dataset
    Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type)

    train_dataset = CustomDataset(Xtrain, Ytrain)
    test_dataset = CustomDataset(Xtest, Ytest)

    # Create data loaders
    batch_size = args.batch_size
    input_dim = len(Xtrain[1])
    encoding_dim = 4000
    model = Autoencoder(input_dim, encoding_dim)
    model.train()
    model = DataParallel(model)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.3)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_log = []
    test_log = []

    # Train the model
    num_epochs = 300
    start_time = time.time()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.to(device)
            outputs = model(batch_X).to(device)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(str(epoch + 1) + ":" + str(train_loss))
        scheduler.step()
        dtime = "%s" % datetime.now()
        log = [epoch, dtime, train_loss]
        train_log.append(log)

    end_time = time.time()
    train_time = end_time - start_time
    print("Training time:", train_time, "seconds.")

    # # Evaluate the model
    # model.eval()
    # mse = 0
    # start_time = time.time()
    #
    # xtrue = []
    # xpred = []
    # # with torch.no_grad():
    # for batch_X, batch_y in test_loader:
    #     batch_X = batch_X.to(device)
    #     xhat = model(batch_X.unsqueeze(1).to(device))
    #     xtrue.extend(batch_X.cpu().numpy())
    #     xpred.extend(xhat.cpu().detach().numpy())
    #     mse += criterion(xhat, batch_X).item()
    #
    # end_time = time.time()
    # test_time = end_time - start_time
    # print("Testing time:", test_time, "seconds")
    # Save the model weights
    os.makedirs("Dataset" + str(args.data_type) + "/AutoEncoder", exist_ok=True)
    torch.save(model.state_dict(), "Dataset" + str(args.data_type) + "/AutoEncoder/model_weights_ebv.pth")

    # # Print the MSE score and R-Squared score
    # mse = mse / len(test_loader)
    # dtime = "%s" % datetime.now()
    # test_log.append([train_time, test_time, dtime, mse])
    # print("MSE score:", mse)
    #
    # # Save the results
    # train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
    # test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score'])
    #
    # train_log.to_csv("Dataset" + str(args.data_type) + "/AutoEncoder/train_log.csv", index=False)
    # test_log.to_csv("Dataset" + str(args.data_type) + "/AutoEncoder/test_log.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--data_type', type=int, default=1, help='4->human simulated, 1,2->pig')
    parser.add_argument('--gpus', type=str, default='1,2,3,4', help='model prefix')
    args = parser.parse_args()

    # Run
    run(args)

