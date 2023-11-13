from sklearn.linear_model import Lasso
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



# 创建LASSO回归模型对象
lasso = Lasso(alpha=1.0)       # 设置正则化参数alpha，默认为1.0

# 拟合模型
lasso.fit(Xtrain, Ytrain)

# 获取模型参数
y_pred = lasso.predict(Xtest) # 使用模型进行预测
mse_result = mse(Ytest, y_pred)
print(mse_result)

test_log = []
time = "%s"%datetime.now()
test_log.append([time, mse_result])

test_log = pd.DataFrame(test_log, columns=['time','MSE score'])

test_log.to_csv("Dataset4/LASSO/test_log.csv",index=False)


