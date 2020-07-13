"""
    train on the drug screening docking data:
    `./dataset/drug_screen/`
"""

from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader, Subset

from NeuralGraph.dataset import MolData, SmileData
from NeuralGraph.model import QSAR, MLP, PreFP
from NeuralGraph.util import dev


FP_METHODS = ["morgan", "nfp"]
FP_LEN = 1 << 10  # fingerprint length for circular FP

# derived from the dataset folder structure
TARGET_CHOICES = ['3CLPro_3', '3CLPro_2', '3CLPro_1', 'CoV_RDB_DA',
                  'CoV_RDB_AB', 'CoV_RDB_CD', 'CoV_RDB_A_1', 'CoV_RDB_BC',
                  'PLPRO_2', 'plpro_1', 'NSP15_3_6w01', 'NSP15_2_6w01',
                  'NSP15_1_6w01', 'DNMT3A_chainA', 'NSUN2_model', 'Mpro-x0104',
                  'Mpro-x0161', 'Mpro-x0305', 'Mpro-x0107', 'DNMT1_chainA',
                  'NSUN6', 'adrp_adpr_A', 'NSP15_2_6vww', 'NSP15_1_6vww']

TARGET_TRAIN = ['3CLPro_2', 'CoV_RDB_CD', '3CLPro_1', 'NSP15_1_6vww',
                'adrp_adpr_A', 'DNMT3A_chainA', 'NSP15_1_6w01', '3CLPro_3',
                'NSP15_2_6vww', 'CoV_RDB_A_1', 'DNMT1_chainA', 'Mpro-x0305',
                'NSUN2_model', 'NSP15_2_6w01', 'Mpro-x0107', 'plpro_1',
                'PLPRO_2', 'Mpro-x0104']

TARGET_TEST = ['CoV_RDB_BC', 'NSUN6', 'CoV_RDB_DA', 'CoV_RDB_AB', 'Mpro-x0161']


def normalize_array(A):
    """ normalize the target score
    """
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
    if len(target_name) == 1:
        # PyTorch MSELoss treats one class differently from multi-class
        y = np.squeeze(y)
    norm_fn, rstr_fn = normalize_array(y)
    target = norm_fn(y)
    data = MolData(x, target, use_tqdm=use_tqdm)
    return data, norm_fn, rstr_fn

def mse(x, y, dim=None):
    return ((x-y)**2).mean(dim)

def create_net(hid_dim, n_class, pre_trained=None):
    """
        pre_trained := the pretrained model file path
    """
    if pre_trained is None:
        # create a new QSAR network
        net = QSAR(hid_dim=128, n_class=n_class)
    else:
        if not Path(pre_trained).exists():
            raise FileNotFoundError
        prenet = torch.load(pre_trained, map_location=dev)
        net = PreFP(prenet.nfp, hid_dim=128, n_class=n_class)

    return net

    return net
def main(args):
    BSZ, RUNS, LR, N_EPOCH = args.batch_size, args.runs, args.lr, args.epochs
    DATAFOLDER = Path(args.datafolder)
    assert DATAFOLDER.exists()
    OUTPUT = args.output_dir+DATAFOLDER.resolve().stem
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    # load data
    NCLASS = len(args.target_name)
    if len(args.target_name) > 1:
        OUTPUT += '_'.join(t for t in args.target_name)
    else:
        OUTPUT += args.target_name[0]
    train_data, norm_fn, rstr_fn =\
        construct_dataset(DATAFOLDER/'train.csv', args.target_name,
                          args.sample, args.use_tqdm)
    val_smiles, val_y = load_csv(DATAFOLDER/'val.csv', args.target_name,
                                 args.sample)
    test_smiles, test_y = load_csv(DATAFOLDER/'test.csv', args.target_name,
                                   args.sample)
    if len(args.target_name) == 1:
        val_y, test_y = np.squeeze(val_y), np.squeeze(test_y)
    val_data = MolData(val_smiles, norm_fn(val_y), args.use_tqdm)
    test_data = MolData(test_smiles, norm_fn(test_y), args.use_tqdm)

    if args.fp_method == FP_METHODS[0]:
        raise NotImplementedError

    res = []
    for _ in range(RUNS):
        train_loader = DataLoader(train_data, batch_size=BSZ,
                                  shuffle=True, drop_last=True,
                                  pin_memory=True)
        valid_loader = DataLoader(val_data, batch_size=BSZ,
                                  shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=BSZ, shuffle=False)
        # net = QSAR(hid_dim=128, n_class=NCLASS)
        net = create_net(hid_dim=128, n_class=NCLASS,
                         pre_trained=args.fine_tune)
        model_path = OUTPUT+"_"
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH,
                      path=model_path,
                      criterion=nn.MSELoss(), lr=LR)
        score = net.predict(test_loader)
        gt = test_y
        prd = rstr_fn(score)
        res.append(mse(gt, prd, 0))
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
    parser.add_argument("--output_dir", help="specify the output directory",
                        type=str, default="./output/")
    parser.add_argument("-b", "--batch_size", help="batch size",
                        default=64, type=int)
    parser.add_argument("-e", "--epochs", help="number of epochs",
                        default=500, type=int)
    parser.add_argument("-r", "--runs", help="number of runs",
                        default=5, type=int)
    parser.add_argument("-l", "--lr", help="learning rate",
                        default=1e-3, type=float)
    parser.add_argument("-f", "--fine_tune", help="load pretrained model",
                        type=str)  # use pre-trained model on new targets
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
