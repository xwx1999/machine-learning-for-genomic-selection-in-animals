import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse

# Read data
X = pd.read_csv('Dataset4/QTLMAS2010ny012.csv',header=None,low_memory=False)
X = X.iloc[:,1:-1]
Y = pd.read_csv('Dataset4/QTLMAS2010ny012.csv',header=None,low_memory=False)
Y = Y.iloc[:,0]


# Split train-test data
Xtrain = X.iloc[:2226].values  # Xtrain
Ytrain = Y.iloc[:2226].values  # ytrain
Xtest = X.iloc[2226:].values  # Xtest
Ytest = Y.iloc[2226:].values  # ytest

Xtrain = Xtrain.astype(np.float32)
Ytrain = Ytrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Ytest = Ytest.astype(np.float32)

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
    w = np.linalg.inv(H) @ X @ Y

    # # 预测值
    # y_pred = X.T @ w

    return w


# 示例数据
X = Xtrain.T  # 基因型矩阵
Y = Ytrain  # 表型值
lambda_reg = 0.1  # 正则化参数

# 调用GBLUP函数进行预测
w = gblup(X, Y, lambda_reg)
y_pred = Xtest @ w

mse_result = mse(Ytest, y_pred)
print(mse_result)

test_log = []
time = "%s"%datetime.now()
test_log.append([time, mse_result])

test_log = pd.DataFrame(test_log, columns=['time','MSE score'])

test_log.to_csv("Dataset4/GBLUP/test_log.csv",index=False)