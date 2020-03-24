from pathlib import Path
from torch.utils.data import DataLoader, Subset
from NeuralGraph.dataset import MolData
from NeuralGraph.model import QSAR
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse

EXP_NAMES = ["solubility", "drug_efficacy", "photovoltaic"]


def split_train_valid_test(n, p=0.8, v=0.1, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    s = int(n*p)
    t = int(n*v)
    # train, valid, test
    return idx[:s], idx[s:(s+t)], idx[(s+t):]


def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    def norm_func(X): return (X-mean) / std
    def restore_func(X): return X * std + mean
    return norm_func, restore_func


def load_csv(data_file, target_name):
    df = pd.read_csv(data_file)
    return df['smiles'], df[target_name].values


def mse(x, y):
    return ((x-y)**2).mean()


def main(args):
    BSZ, RUNS, LR, N_EPOCH = args.batch_size, args.runs, args.lr, args.epochs
    OUTPUT, TGT_COL_NAME = [None]*2
    if args.experiment == EXP_NAMES[0]:
        OUTPUT = './output/best_delaney.pkl'
        DATAFILE = Path('./dataset/solubility/delaney-processed.csv')
        TGT_COL_NAME = 'measured log solubility in mols per litre'
    elif args.experiment == EXP_NAMES[1]:
        OUTPUT = './output/best_efficacy.pkl'
        DATAFILE = Path('./dataset/drug_efficacy/malaria-processed.csv')
        TGT_COL_NAME = 'activity'
    elif args.experiment == EXP_NAMES[2]:
        OUTPUT = './output/best_photovoltaic.pkl'
        DATAFILE = Path('./dataset/photovoltaic_efficiency/cep-processed.csv')
        TGT_COL_NAME = 'PCE'
    else:
        raise NotImplementedError

    res = []
    for _ in range(RUNS):
        input_data, target = load_csv(DATAFILE, TGT_COL_NAME)
        train_idx, valid_idx, test_idx = split_train_valid_test(len(target),
                                                                seed=None)
        norm_func, restore_func = normalize_array(
            np.concatenate([target[train_idx], target[valid_idx]], axis=0))
        target = norm_func(target)
        data = MolData(input_data, target)
        train_loader = DataLoader(Subset(data, train_idx), batch_size=BSZ,
                                  shuffle=True, drop_last=True)
        valid_loader = DataLoader(Subset(data, valid_idx), batch_size=BSZ,
                                  shuffle=False)
        test_loader = DataLoader(Subset(data, test_idx), batch_size=BSZ,
                                 shuffle=False)
        net = QSAR(hid_dim=128, n_class=1)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path=OUTPUT,
                      criterion=nn.MSELoss(), lr=LR)
        score = net.predict(test_loader)
        gt = restore_func(target[test_idx])
        prd = restore_func(score)
        res.append(mse(gt, prd))
        print(mse(gt,prd))

    avg_mse, std_mse = np.asarray(res).mean(), np.asarray(res).std()
    return avg_mse, std_mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", default="solubility", type=str,
                        help="Specify the experiment name",
                        choices=EXP_NAMES)
    parser.add_argument("-b", "--batch-size", help="batch size",
                        default=128, type=int)
    parser.add_argument("-e", "--epochs", help="number of epochs",
                        default=200, type=int)
    parser.add_argument("-r", "--runs", help="number of runs",
                        default=5, type=int)
    parser.add_argument("-l", "--lr", help="learning rate",
                        default=1e-3, type=float)
    parsed_args = parser.parse_args()
    print(main(parsed_args))
