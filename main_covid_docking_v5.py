"""
    train on the drug screening docking data:
    `./dataset/drug_screen/`
"""

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from NeuralGraph.dataset import MolData, SmileData
from NeuralGraph.model import QSAR, MLP
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse


FP_METHODS = ["morgan", "nfp"]
FP_LEN = 1 << 10  # fingerprint length for circular FP

# derived from the dataset folder structure
TARGET_CHOICES = ['3CLPro_3', '3CLPro_2', '3CLPro_1', 'CoV_RDB_DA', 'CoV_RDB_AB',
                  'CoV_RDB_CD', 'CoV_RDB_A_1', 'CoV_RDB_BC', 'PLPRO_2', 'plpro_1',
                  'NSP15_3_6w01', 'NSP15_2_6w01', 'NSP15_1_6w01', 'DNMT3A_chainA',
                  'NSUN2_model', 'Mpro-x0104', 'Mpro-x0161', 'Mpro-x0305', 'Mpro-x0107',
                  'DNMT1_chainA', 'NSUN6', 'adrp_adpr_A', 'NSP15_2_6vww', 'NSP15_1_6vww']


def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    def norm_func(X): return (X-mean) / std
    def restore_func(X): return X * std + mean
    return norm_func, restore_func


def load_csv(data_file, target_name, sample=None):
    """ the key is "SMILES", and target_name should match the column name
        target_name is label or list of labels
    """
    df = pd.read_csv(data_file, index_col="SMILES")
    if sample is not None:
        df = df.sample(sample) if isinstance(sample, int) else df.sample(frac=sample)
    return list(df.index), df[target_name].values

def construct_dataset(data_file, target_name, sample=None, use_tqdm=None):
    x, y = load_csv(data_file, target_name, sample)
    norm_fn, rstr_fn = normalize_array(y)
    target = norm_fn(y)
    data = MolData(x, target, use_tqdm=use_tqdm)
    return data, norm_fn, rstr_fn

def mse(x, y, dim=None):
    return ((x-y)**2).mean(dim)


def main(args):
    BSZ, RUNS, LR, N_EPOCH = args.batch_size, args.runs, args.lr, args.epochs
    DATAFOLDER = Path(args.datafolder)
    assert DATAFOLDER.exists()
    OUTPUT = args.output_dir+DATAFOLDER.resolve().stem

    # load data
    train_data, norm_fn, rstr_fn =\
        construct_dataset(DATAFOLDER/'train.csv', args.target_name,
                          args.sample, args.use_tqdm)
    val_smiles, val_y = load_csv(DATAFOLDER/'val.csv', args.target_name,
                                 args.sample)
    test_smiles, test_y = load_csv(DATAFOLDER/'test.csv', args.target_name,
                                   args.sample)
    val_data = MolData(val_smiles, norm_fn(val_y), args.use_tqdm)
    test_data = MolData(test_smiles, norm_fn(test_y), args.use_tqdm)
    NCLASS = len(args.target_name)
    OUTPUT += args.target_name

    if args.fp_method == FP_METHODS[0]:
        raise NotImplementedError
    #def build_data_net(args, target):
    #    if args.fp_method == FP_METHODS[0]:
    #        #""" CFP """
    #        data = SmileData(SMILES, target, fp_len=FP_LEN, radius=4)
    #        net = lambda : MLP(hid_dim=FP_LEN, n_class=NCLASS)
    #        return data, net
    #    elif args.fp_method == FP_METHODS[1]:
    #        #""" NFP """
    #        net = lambda : QSAR(hid_dim=128, n_class=NCLASS)
    #        data = MolData(SMILES, target, use_tqdm=args.use_tqdm)
    #        return data, net
    #    else:
    #        raise NotImplementedError

    res = []
    for _ in range(RUNS):
        train_loader = DataLoader(train_data, batch_size=BSZ,
                                  shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(val_data, batch_size=BSZ,
                                  shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=BSZ, shuffle=False)
        net = QSAR(hid_dim=128, n_class=NCLASS)
        model_path = OUTPUT+str(_)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH,
                      path=model_path,
                      criterion=nn.MSELoss(), lr=LR)
        score = net.predict(test_loader)
        gt = test_y
        prd = rstr_fn(score)
        res.append(mse(gt, prd))
        print(f"mse_{DATAFOLDER.stem}_RUN_{_}: {mse(gt, prd)}")
        print(f"mse_percls_{DATAFOLDER.stem}_RUN_{_}: {mse(gt, prd, 0)}")

    avg_mse, std_mse = np.asarray(res).mean(), np.asarray(res).std()
    return avg_mse, std_mse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datafolder", type=str,
                        help="Specify the input smi filename")
    parser.add_argument("fp_method", default="nfp", type=str,
                        help="Specify the fingerprint method",
                        choices=FP_METHODS)
    parser.add_argument("--delimiter", help="choose the delimiter of the smi\
                        file", type=str, default=",")
    parser.add_argument("--output_dir", help="specify the output directory",
                        type=str, default="./output/")
    parser.add_argument("-b", "--batch-size", help="batch size",
                        default=64, type=int)
    parser.add_argument("-e", "--epochs", help="number of epochs",
                        default=500, type=int)
    parser.add_argument("-r", "--runs", help="number of runs",
                        default=5, type=int)
    parser.add_argument("-l", "--lr", help="learning rate",
                        default=1e-3, type=float)
    parser.add_argument("--target_name", nargs='+',
                        choices=TARGET_CHOICES,
                        help="specify the column name(s)")
    parser.add_argument("--sample", help="train on a sample of the dataset",
                        type=int)
    parser.add_argument("--use_tqdm", action="store_true",
                        help="show progress bar")
    parsed_args = parser.parse_args()
    print("#",parsed_args)
    res = main(parsed_args)
    print(f"{Path(parsed_args.datafolder).stem}: {res[0]:.4f}, {res[1]:.4f}")
