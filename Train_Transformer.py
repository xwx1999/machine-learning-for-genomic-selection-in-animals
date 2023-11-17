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


    # Dataset(not for Transformer)
    batch_size = args.batch_size
    PCA_mse = []
    PCA_r_sqr = []
    ICA_mse = []
    ICA_r_sqr = []
    TSNE_mse = []
    TSNE_r_sqr = []
    RT_mse = []
    RT_r_sqr = []



    if args.data_type != 4:
        for i in tqdm(range(5)):
            #PCA
            Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type, "PCA")

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
            os.makedirs("Dataset"+str(args.data_type)+"/Transformer_PCA", exist_ok=True)
            torch.save(model.state_dict(), "Dataset"+str(args.data_type)+"/Transformer_PCA/model_weights_ebv.pth")

            # Print the MSE score and R-Squared score
            r_squared = r2_score(ytrue, ypred)
            mse = mse / len(test_loader)
            PCA_mse.append(mse)
            PCA_r_sqr.append(r_squared)
            dtime = "%s" % datetime.now()
            test_log.append([train_time, test_time, dtime, mse, r_squared])
            print("MSE score:", mse)
            print("R-squared score:", r_squared)

            # Save the results
            train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
            test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

            train_log.to_csv("Dataset"+str(args.data_type)+"/Transformer_PCA/train_log.csv", index=False)
            test_log.to_csv("Dataset"+str(args.data_type)+"/Transformer_PCA/test_log.csv", index=False)

            # ICA
            Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type, "ICA")

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
            os.makedirs("Dataset" + str(args.data_type) + "/Transformer_ICA", exist_ok=True)
            torch.save(model.state_dict(), "Dataset" + str(args.data_type) + "/Transformer_ICA/model_weights_ebv.pth")

            # Print the MSE score and R-Squared score
            r_squared = r2_score(ytrue, ypred)
            mse = mse / len(test_loader)
            ICA_mse.append(mse)
            ICA_r_sqr.append(r_squared)
            dtime = "%s" % datetime.now()
            test_log.append([train_time, test_time, dtime, mse, r_squared])
            print("MSE score:", mse)
            print("R-squared score:", r_squared)

            # Save the results
            train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
            test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

            train_log.to_csv("Dataset" + str(args.data_type) + "/Transformer_ICA/train_log.csv", index=False)
            test_log.to_csv("Dataset" + str(args.data_type) + "/Transformer_ICA/test_log.csv", index=False)

            # TSNE
            Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type, "TSNE")

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
            os.makedirs("Dataset" + str(args.data_type) + "/Transformer_TSNE", exist_ok=True)
            torch.save(model.state_dict(), "Dataset" + str(args.data_type) + "/Transformer_TSNE/model_weights_ebv.pth")

            # Print the MSE score and R-Squared score
            r_squared = r2_score(ytrue, ypred)
            mse = mse / len(test_loader)
            TSNE_mse.append(mse)
            TSNE_r_sqr.append(r_squared)
            dtime = "%s" % datetime.now()
            test_log.append([train_time, test_time, dtime, mse, r_squared])
            print("MSE score:", mse)
            print("R-squared score:", r_squared)

            # Save the results
            train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
            test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

            train_log.to_csv("Dataset" + str(args.data_type) + "/Transformer_TSNE/train_log.csv", index=False)
            test_log.to_csv("Dataset" + str(args.data_type) + "/Transformer_TSNE/test_log.csv", index=False)

        PCA_mse_mean, PCA_mse_std = np.mean(PCA_mse), np.std(PCA_mse)
        PCA_r_mean, PCA_r_std = np.mean(PCA_r_sqr),np.std(PCA_r_sqr)
        print("PCA:")
        print(PCA_mse_mean, PCA_mse_std)
        print(PCA_r_mean, PCA_r_std)
        print('-'*100)

        ICA_mse_mean, ICA_mse_std = np.mean(ICA_mse), np.std(ICA_mse)
        ICA_r_mean, ICA_r_std = np.mean(ICA_r_sqr), np.std(ICA_r_sqr)
        print("ICA:")
        print(ICA_mse_mean, ICA_mse_std)
        print(ICA_r_mean, ICA_r_std)
        print('-' * 100)


        TSNE_mse_mean, TSNE_mse_std = np.mean(TSNE_mse), np.std(TSNE_mse)
        TSNE_r_mean, TSNE_r_std = np.mean(TSNE_r_sqr), np.std(TSNE_r_sqr)
        print("TSNE:")
        print(TSNE_mse_mean, TSNE_mse_std)
        print(TSNE_r_mean, TSNE_r_std)
        print('-' * 100)

        log = {'PCA': [PCA_mse_mean, PCA_mse_std,PCA_r_mean,PCA_r_std],
               'ICA': [ICA_mse_mean,ICA_mse_std, ICA_r_mean, ICA_r_std],
               'TSNE': [TSNE_mse_mean, TSNE_mse_std, TSNE_r_mean, TSNE_r_std]
               }
        log = pd.DataFrame(log)
        log.to_csv("Dataset" + str(args.data_type) + "/log_transformer.csv")

    elif args.data_type == 4:
        for i in tqdm(range(5)):
            # RT
            Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type, None)


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
            optimizer = optim.Adam(model.parameters(), lr=0.1)
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
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(train_loss)
                scheduler.step()
                dtime = "%s" % datetime.now()
                log = [epoch, dtime, train_loss]
                train_log.append(log)

            end_time = time.time()
            train_time = end_time - start_time
            print("Training time:", train_time, "seconds.")

            # Evaluate the model
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
            os.makedirs("Dataset" + str(args.data_type) + "/Transformer_PCA", exist_ok=True)
            torch.save(model.state_dict(), "Dataset" + str(args.data_type) + "/Transformer_PCA/model_weights_ebv.pth")

            # Print the MSE score and R-Squared score
            print(np.isnan(ytrue).any())
            print(np.isnan(ypred).any())
            r_squared = r2_score(ytrue, ypred)
            mse = mse / len(test_loader)
            RT_mse.append(mse)
            RT_r_sqr.append(r_squared)
            dtime = "%s" % datetime.now()
            test_log.append([train_time, test_time, dtime, mse, r_squared])
            print("MSE score:", mse)
            print("R-squared score:", r_squared)

            # Save the results
            train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
            test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

            train_log.to_csv("Dataset" + str(args.data_type) + "/Transformer_PCA/train_log.csv", index=False)
            test_log.to_csv("Dataset" + str(args.data_type) + "/Transformer_PCA/test_log.csv", index=False)

        RT_mse_mean, RT_mse_std = np.mean(RT_mse), np.std(RT_mse)
        RT_r_mean, RT_r_std = np.mean(RT_r_sqr), np.std(RT_r_sqr)
        print("RT:")
        print(RT_mse_mean, RT_mse_std)
        print(RT_r_mean, RT_r_std)
        print('-' * 100)

        log = {'RT': [RT_mse_mean, RT_mse_std, RT_r_mean, RT_r_std]}
        log = pd.DataFrame(log)
        log.to_csv("Dataset" + str(args.data_type) + "/log_transformer.csv")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--data_type', type=int, default=1, help='1->human simulated, 2,4->pig')
    parser.add_argument('--reduction_type', type=str, default=None, help='PCA or TSNE or ICA or AE')
    parser.add_argument('--gpus', type=str, default='1,2,3,4', help='model prefix')
    args = parser.parse_args()

    # Run
    run(args)

