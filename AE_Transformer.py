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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset(not for Transformer)
    batch_size = args.batch_size
    AET_mse = []
    AET_r_sqr = []

    for i in tqdm(range(5)):

        Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type)

        # AE
        input_dim = Xtrain.shape[1]
        encoding_dim = 4000
        ae = Autoencoder(input_dim, encoding_dim)
        weights = torch.load("Dataset" + str(args.data_type) + "/AutoEncoder/model_weights_ebv.pth", map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
        ae.load_state_dict(new_state_dict)
        ae.eval()
        with torch.no_grad():
            Xtrain = ae.encoder(torch.tensor(Xtrain, dtype=torch.float32))
            Xtest = ae.encoder(torch.tensor(Xtest, dtype=torch.float32))

        train_dataset = CustomDataset(Xtrain, Ytrain)
        test_dataset = CustomDataset(Xtest, Ytest)

        # Create data loaders
        input_size = len(train_dataset[0][0])
        hidden_size = 128
        model = TransformerRegressor(input_size, hidden_size)
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
                batch_y = batch_y.to(device)
                outputs = model(batch_X.unsqueeze(1)).to(device)
                loss = criterion(outputs, batch_y.unsqueeze(1))
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

        # Evaluate the model
        ae.eval()
        model.eval()
        mse = 0
        start_time = time.time()

        ytrue = []
        ypred = []
        # with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            yhat = model(batch_X.unsqueeze(1).to(device))
            ytrue.extend(batch_y.cpu().numpy())
            ypred.extend(yhat.cpu().detach().numpy())
            mse += criterion(yhat, batch_y.unsqueeze(1)).item()

        end_time = time.time()
        test_time = end_time - start_time
        print("Testing time:", test_time, "seconds")
        # Save the model weights
        os.makedirs("Dataset"+str(args.data_type)+"/AE_T", exist_ok=True)
        torch.save(model.state_dict(), "Dataset"+str(args.data_type)+"/AE_T/model_weights_ebv.pth")

        # Print the MSE score and R-Squared score
        r_squared = r2_score(ytrue, ypred)
        mse = mse / len(test_loader)
        AET_mse.append(mse)
        AET_r_sqr.append(r_squared)
        dtime = "%s" % datetime.now()
        test_log.append([train_time, test_time, dtime, mse, r_squared])
        print("MSE score:", mse)
        print("R-squared score:", r_squared)

        # Save the results
        train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
        test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

        train_log.to_csv("Dataset"+str(args.data_type)+"/AE_T/train_log.csv", index=False)
        test_log.to_csv("Dataset"+str(args.data_type)+"/AE_T/test_log.csv", index=False)

    AET_mse_mean, AET_mse_std = np.mean(AET_mse), np.std(AET_mse)
    AET_r_mean, AET_r_std = np.mean(AET_r_sqr), np.std(AET_r_sqr)
    print("AET:")
    print(AET_mse_mean, AET_mse_std)
    print(AET_r_mean, AET_r_std)
    print('-' * 100)

    log = {'AET': [AET_mse_mean, AET_mse_std, AET_r_mean, AET_r_std]}
    log = pd.DataFrame(log)
    log.to_csv("Dataset" + str(args.data_type) + "/log_AET.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--data_type', type=int, default=1, help='1->human simulated, 2,4->pig')
    parser.add_argument('--reduction_type', type=str, default=None, help='PCA or TSNE or ICA or AE')
    parser.add_argument('--gpus', type=str, default='1,2,3,4', help='model prefix')
    args = parser.parse_args()

    # Run
    run(args)