# 导入所需的库
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse

# Read data
X = pd.read_csv('Dataset2/genotype.csv',header=None,low_memory=True)
X = X.iloc[1:,2:]
Y = pd.read_csv('Dataset2/mortality_EBV.csv',header=None)
Y = Y.iloc[2:,1]

# Split train-test data
Xtrain = X.iloc[:1000].values  # Xtrain
Ytrain = Y.iloc[:1000].values  # ytrain
Xtest = X.iloc[1000:].values  # Xtest
Ytest = Y.iloc[1000:].values  # ytest

Xtrain = Xtrain.astype(np.float32)
Ytrain = Ytrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Ytest = Ytest.astype(np.float32)



# 将数据转换为LightGBM所需的数据格式
train_data = lgb.Dataset(Xtrain, label=Ytrain)

# 设置模型参数
params = {
    'boosting_type': 'gbdt',         # 使用梯度提升树算法
    'objective': 'regression',       # 回归任务
    'metric': 'rmse',                # 使用均方根误差评估模型性能
    'num_leaves': 31,                # 每棵树的叶子节点数目
    'learning_rate': 0.05,           # 学习率
    'feature_fraction': 0.9,         # 每次迭代时随机选择特征的比例
    'bagging_fraction': 0.8,         # 每次迭代时随机选择数据的比例
    'bagging_freq': 5,               # bagging的频率
    'verbose': 0                     # 控制训练过程中输出的信息
}

# 训练模型
model = lgb.train(params, train_data, num_boost_round=100)

# 使用训练好的模型进行预测
y_pred = model.predict(Xtest)

# 输出预测结果
mse_result = mse(Ytest, y_pred)
print(mse_result)

test_log = []
time = "%s"%datetime.now()
test_log.append([time, mse_result])

test_log = pd.DataFrame(test_log, columns=['time','MSE score'])

test_log.to_csv("Dataset2/LGBM/test_log.csv",index=False)