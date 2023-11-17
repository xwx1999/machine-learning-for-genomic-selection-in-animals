import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from tqdm import tqdm
import torch.nn as nn
from sklearn.linear_model import LinearRegression
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
from sklearn.metrics import mean_squared_error as MSE
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
    GBLUP_mse = []
    GBLUP_r_sqr = []
    LASSO_mse = []
    LASSO_r_sqr = []
    # LGBM_mse = []
    # LGBM_r_sqr = []
    LSTM_mse = []
    LSTM_r_sqr = []
    CNN_mse = []
    CNN_r_sqr = []


    for i in tqdm(range(5)):
        Xtrain, Ytrain, Xtest, Ytest = SNPDataset(args.data_type, args.reduction_type)

        # GBLUP
        start_time = time.time()
        allele_frequencies = Xtrain.mean(axis=0)
        variance = 2 * allele_frequencies * (1 - allele_frequencies)
        variance = (variance - min(variance)) / max(variance) - min(variance)
        standardized_Xtrain = (Xtrain - allele_frequencies) / np.sqrt(variance)
        std_Xtrain = pd.DataFrame(standardized_Xtrain)
        standardized_Xtrain = std_Xtrain.fillna(std_Xtrain.mean()).values

        model = LinearRegression()
        model.fit(standardized_Xtrain, Ytrain)
        end_time = time.time()
        train_time = end_time - start_time
        start_time = time.time()

        # 在测试集上进行预测
        allele_frequencies = Xtest.mean(axis=0)
        variance = 2 * allele_frequencies * (1 - allele_frequencies)
        variance = (variance - min(variance)) / max(variance) - min(variance)
        standardized_Xtest = (Xtest - allele_frequencies) / np.sqrt(variance)
        Ypred = model.predict(standardized_Xtest)
        end_time = time.time()
        test_time = end_time - start_time
        mse_result = MSE(Ytest, Ypred)
        r_squared = r2_score(Ytest, Ypred)
        GBLUP_mse.append(mse_result)
        GBLUP_r_sqr.append(r_squared)
        print(mse_result)
        print(r_squared)

        test_log = []
        test_log.append([train_time, test_time, mse_result, r_squared])
        test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'MSE score', 'R-squared'])
        os.makedirs("Dataset"+str(args.data_type)+"/GBLUP", exist_ok=True)
        test_log.to_csv("Dataset"+str(args.data_type)+"/GBLUP/test_log"+str(i)+".csv", index=False)


        #LASSO
        start_time = time.time()
        lasso = Lasso(alpha=1.0)
        lasso.fit(Xtrain, Ytrain)
        end_time = time.time()
        train_time = end_time - start_time
        start_time = time.time()
        Ypred = lasso.predict(Xtest)
        end_time = time.time()
        test_time = end_time - start_time
        mse_result = MSE(Ytest, Ypred)
        r_squared = r2_score(Ytest, Ypred)
        LASSO_mse.append(mse_result)
        LASSO_r_sqr.append(r_squared)
        print(mse_result)
        print(r_squared)

        test_log = []
        test_log.append([train_time, test_time, mse_result, r_squared])
        test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'MSE score', 'R-squared'])
        os.makedirs("Dataset" + str(args.data_type) + "/LASSO", exist_ok=True)
        test_log.to_csv("Dataset" + str(args.data_type) + "/LASSO/test_log_" + str(i) + ".csv", index=False)


        # #LGBM
        # # 将数据转换为LightGBM所需的数据格式
        # train_data = lgb.Dataset(Xtrain, label=Ytrain)
        #
        # # 设置模型参数
        # params = {
        #     'boosting_type': 'gbdt',  # 使用梯度提升树算法
        #     'objective': 'regression',  # 回归任务
        #     'metric': 'rmse',  # 使用均方根误差评估模型性能
        #     'num_leaves': 31,  # 每棵树的叶子节点数目
        #     'learning_rate': 0.05,  # 学习率
        #     'feature_fraction': 0.9,  # 每次迭代时随机选择特征的比例
        #     'bagging_fraction': 0.8,  # 每次迭代时随机选择数据的比例
        #     'bagging_freq': 5,  # bagging的频率
        #     'verbose': 0  # 控制训练过程中输出的信息
        # }
        # start_time = time.time()
        # # 训练模型
        # model = lgb.train(params, train_data, num_boost_round=100)
        # end_time = time.time()
        # train_time = end_time - start_time
        # start_time = time.time()
        # # 使用训练好的模型进行预测
        # Ypred = model.predict(Xtest)
        # end_time = time.time()
        # test_time = end_time - start_time
        #
        # mse_result = mse(Ytest, Ypred)
        # r_squared = r2_score(Ytest, Ypred)
        # LGBM_mse_mse.append(mse_result)
        # LGBM_r_sqr.append(r_squared)
        # print(mse_result)
        # print(r_squared)
        #
        # test_log = []
        # test_log.append([train_time, test_time, mse_result, r_squared])
        # test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'MSE score', 'R-squared'])
        # os.makedirs("Dataset" + str(args.data_type) + "/LGBM", exist_ok=True)
        # test_log.to_csv("Dataset" + str(args.data_type) + "/LGBM/test_log_" + str(i) + ".csv", index=False)




        #LSTM

        # Convert input to PyTorch format and create datasets
        train_dataset = CustomDataset(Xtrain, Ytrain)
        test_dataset = CustomDataset(Xtest, Ytest)


        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = len(Xtrain[0])
        hidden_size = batch_size
        output_size = 1

        model = LSTMModel(input_size, hidden_size, output_size)
        model.train()
        model = DataParallel(model).to(device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, 100, 0.3)


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
                outputs = model(batch_X.unsqueeze(1))
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

        Ytrue = []
        Ypred = []
        # with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            yhat = model(batch_X.unsqueeze(1)).to(device)

            Ytrue.extend(batch_y.cpu().numpy())
            Ypred.extend(yhat.cpu().detach().numpy())
            mse += criterion(yhat, batch_y.unsqueeze(1)).item()

        end_time = time.time()
        test_time = end_time - start_time
        print("Testing time:", test_time, "seconds")
        # Save the model weights
        os.makedirs("Dataset"+str(args.data_type)+"/LSTM", exist_ok=True)
        torch.save(model.state_dict(), "Dataset"+str(args.data_type)+"/LSTM/model_weights_ebv_"+str(i)+".pth")

        r_squared = r2_score(Ytrue, Ypred)
        LSTM_r_sqr.append(r_squared)
        mse = mse / len(test_loader)
        LSTM_mse.append(mse)
        dtime = "%s" % datetime.now()
        test_log.append([train_time, test_time, dtime, mse, r_squared])
        print("MSE score:", mse)
        print("R-squared score:", r_squared)

        # Save the results
        train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
        test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

        train_log.to_csv("Dataset"+str(args.data_type)+"/LSTM/train_log"+str(i)+".csv", index=False)
        test_log.to_csv("Dataset"+str(args.data_type)+"/LSTM/test_log"+str(i)+".csv", index=False)




        #CNNGWP
        filter = 50
        kernel = 25
        lambda_reg = 0.5

        # Create the CNNGWP model
        model = CNNGWP(filter, kernel, lambda_reg, args.data_type)
        model.apply(init_weights)
        model.train()
        model = DataParallel(model).to(device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, 100, 0.5)

        # # test
        # example = torch.rand((1, 52842))
        # example = example.unsqueeze(0)  # 将example的维度调整为3维
        # print(model(example).size())

        train_log = []
        test_log = []

        start_time = time.time()
        # Train the model
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
            t = "%s" % datetime.now()
            log = [epoch, t, loss.item()]
            train_log.append(log)

        end_time = time.time()
        train_time = end_time - start_time
        print("Training time:", train_time, "seconds.")
        # Evaluate the model
        model.eval()
        mse = 0
        start_time = time.time()

        Ytrue = []
        Ypred = []

        # with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            yhat = model(batch_X.unsqueeze(1).to(device))
            Ytrue.extend(batch_y.cpu().numpy())
            Ypred.extend(yhat.cpu().detach().numpy())
            mse += criterion(yhat, batch_y.unsqueeze(1)).item()
        end_time = time.time()
        test_time = end_time - start_time
        print("Testing time:", test_time, "seconds")
        # Save the model weights
        os.makedirs("Dataset" + str(args.data_type) + "/CNN", exist_ok=True)
        torch.save(model.state_dict(), "Dataset" + str(args.data_type) + "/CNN/model_weights_ebv_" + str(i) + ".pth")

        r_squared = r2_score(Ytrue, Ypred)
        CNN_r_sqr.append(r_squared)
        mse = mse / len(test_loader)
        CNN_mse.append(mse)
        dtime = "%s" % datetime.now()
        test_log.append([train_time, test_time, dtime, mse, r_squared])
        print("MSE score:", mse)
        print("R-squared score:", r_squared)

        # Save the results
        train_log = pd.DataFrame(train_log, columns=['epoch', 'time', 'train loss'])
        test_log = pd.DataFrame(test_log, columns=['train_time', 'test_time', 'time', 'MSE score', 'R-squared'])

        train_log.to_csv("Dataset" + str(args.data_type) + "/CNN/train_log" + str(i) + ".csv", index=False)
        test_log.to_csv("Dataset" + str(args.data_type) + "/CNN/test_log" + str(i) + ".csv", index=False)

    GBLUP_mse_mean, GBLUP_mse_std = np.mean(GBLUP_mse), np.std(GBLUP_mse)
    GBLUP_r_mean, GBLUP_r_std = np.mean(GBLUP_r_sqr), np.std(GBLUP_r_sqr)
    print("GBLUP:")
    print(GBLUP_mse_mean, GBLUP_mse_std)
    print(GBLUP_r_mean, GBLUP_r_std)
    print('-'*100)

    LASSO_mse_mean, LASSO_mse_std = np.mean(LASSO_mse), np.std(LASSO_mse)
    LASSO_r_mean, LASSO_r_std = np.mean(LASSO_r_sqr), np.std(LASSO_r_sqr)
    print("LASSO:")
    print(LASSO_mse_mean, LASSO_mse_std)
    print(LASSO_r_mean, LASSO_r_std)
    print('-' * 100)

    # LGBM_mse_mean, LGBM_mse_std = np.std(LGBM_mse), np.std(LGBM_mse)
    # LGBM_r_mean, LGBM_r_std = np.std(LGBM_r_sqr), np.std(LGBM_r_sqr)
    # print("LGBM:")
    # print(LGBM_mse_mean, LGBM_mse_std)
    # print(LGBM_r_mean, LGBM_r_std)
    # print('-' * 100)

    LSTM_mse_mean, LSTM_mse_std = np.mean(LSTM_mse), np.std(LSTM_mse)
    LSTM_r_mean, LSTM_r_std = np.mean(LSTM_r_sqr), np.std(LSTM_r_sqr)
    print("LSTM:")
    print(LSTM_mse_mean, LSTM_mse_std)
    print(LSTM_r_mean, LSTM_r_std)
    print('-' * 100)

    CNN_mse_mean, CNN_mse_std = np.mean(CNN_mse), np.std(CNN_mse)
    CNN_r_mean, CNN_r_std = np.mean(CNN_r_sqr), np.std(CNN_r_sqr)
    print("CNN:")
    print(CNN_mse_mean, CNN_mse_std)
    print(CNN_r_mean, CNN_r_std)
    print('-' * 100)

    log = {'GBLUP':[GBLUP_mse_mean, GBLUP_mse_std,GBLUP_r_mean,GBLUP_r_std],
           'LASSO': [LASSO_mse_mean, LASSO_mse_std, LASSO_r_mean, LASSO_r_std],
           'LSTM': [LSTM_mse_mean, LSTM_mse_std, LSTM_r_mean, LSTM_r_std],
           'CNN': [CNN_mse_mean, CNN_mse_std, CNN_r_mean, CNN_r_std]
           }
    log = pd.DataFrame(log)
    log.to_csv("Dataset" + str(args.data_type) + "/log.csv")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--data_type', type=int, default=1, help='1->human simulated, 2,4->pig')
    parser.add_argument('--reduction_type', type=str, default=None, help='PCA or TSNE or ICA or AE')
    parser.add_argument('--gpus', type=str, default='1,2,3,4', help='model prefix')
    args = parser.parse_args()

    # Run
    run(args)

