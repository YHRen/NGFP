from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from NeuralGraph.dataset import MolData
from NeuralGraph.model import QSAR
import torch.nn as nn
import pandas as pd
import numpy as np


def split_train_valid(n, p=0.8, seed=None):
    if seed: np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    s = int(n*p)
    return idx[:s], idx[s:]


def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    A_normed = (A - mean) / std
    def restore_function(X):
        return X * std + mean

    return A_normed, restore_function

def mse(x, y):
    return ((x-y)**2).mean()

if __name__ == '__main__':
    BSZ = 128
    N_EPOCH = 1000
    LR = 1e-4
    DATAFILE = Path('./dataset/solubility/delaney-processed.csv')
    OUTPUT = './output/best_delaney.pkl'
    df = pd.read_csv(DATAFILE)

    target = df['measured log solubility in mols per litre'].values
    target, restore = normalize_array(target)
    data = MolData(df['smiles'], target)
    train_idx, valid_idx = split_train_valid(len(data), p=0.8, seed=7)
    train_loader = DataLoader(Subset(data, train_idx), batch_size=BSZ, \
                              shuffle=True, drop_last=True)
    valid_loader = DataLoader(Subset(data, valid_idx), batch_size=BSZ, \
                              shuffle=False)
    net = QSAR(hid_dim=128, n_class=1)
    net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path=OUTPUT,
                  criterion=nn.MSELoss(), lr=LR)

    score = net.predict(valid_loader)
    gt = restore(target[valid_idx])
    prd = restore(score)
    res = np.concatenate([gt[...,None], prd[...,None], score[...,None]], axis=1)
    print(mse(gt, prd))
    np.savetxt('res.txt', res)
    
