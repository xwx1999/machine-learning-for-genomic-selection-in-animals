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
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import r2_score
from model.backbone import Autoencoder

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


def SNPDataset(dataset_type, reduction=None):
    if dataset_type == 1:
        root_X = 'Dataset1/FileS1/genotypes.txt'
        root_Y = 'Dataset1/FileS1/ebvs.txt'
        X_row = 1
        X_column = 2
        Y_row = 1
        Y_column = 1

    elif dataset_type == 2:
        root_X = 'Dataset2/genotype.csv'
        root_Y = 'Dataset2/mortality_EBV.csv'
        X_row = 1
        X_column = 2
        Y_row = 2
        Y_column = 1

    elif dataset_type == 4:
        root_X = 'Dataset4/simulated.csv'
        root_Y = 'Dataset4/simulated.csv'
        X_row = 1
        X_column = 1
        Y_row = 0
        Y_column = 0


    X = pd.read_csv(root_X, header=None, low_memory=False)
    Y = pd.read_csv(root_Y, header=None, low_memory=False)
    X = X.iloc[X_row:, X_column:]
    Y = Y.iloc[Y_row:, Y_column]
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    df = pd.concat([X, Y], axis=1)
    df = df.sample(frac=1)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    num_train = math.floor(X.shape[0]*0.8)

    Xtrain = X.iloc[:num_train]
    Xtrain = Xtrain.fillna(Xtrain.mean()).values
    Ytrain = Y.iloc[:num_train].values
    Xtest = X.iloc[num_train:]
    Xtest = Xtest.fillna(Xtest.mean()).values
    Ytest = Y.iloc[num_train:].values

    if dataset_type == 4:
        Ytrain = (Ytrain-Ytrain.mean())/Ytrain.std()
        Ytest = (Ytest - Ytest.mean()) / Ytest.std()

    if reduction == "PCA":
        pca = PCA(n_components = Xtest.shape[0])
        Xtrain, Xtest = pca.fit_transform(Xtrain), pca.fit_transform(Xtest)
    elif reduction == "ICA":
        ica = FastICA(n_components = Xtest.shape[0])
        Xtrain, Xtest = ica.fit_transform(Xtrain), ica.fit_transform(Xtest)
    elif reduction == "TSNE":
        tsne = TSNE(n_components=3)
        Xtrain, Xtest = tsne.fit_transform(Xtrain), tsne.fit_transform(Xtest)


    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    Ytest = Ytest.astype(np.float32)

    return Xtrain,Ytrain,Xtest,Ytest

SNPDataset(1,"AE")
