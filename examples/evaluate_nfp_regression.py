import torch
import argparse
import hashlib
import pandas as pd
import numpy as np
import itertools as its
from tabulate import tabulate
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from pathlib import Path, PurePath
from warnings import warn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import (
    r2_score,
    mean_absolute_error as mae,
    mean_squared_error as mse
)
from scipy.stats import pearsonr
try:
    import NeuralGraph
except:
    import sys
    sys.path.insert(1,str(PurePath(Path.cwd()).parent))
    sys.path.insert(1,str(PurePath(Path.cwd())))
from NeuralGraph.dataset import MolData
from NeuralGraph.util import dev

BSZ = 32  # batch_size
SHUFFLE_SIG = None  # random shuffle signature
def split_train_valid_test(n, p=0.8, v=0.1, seed=None):
    global SHUFFLE_SIG
    if seed:
        np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    s = int(n*p)
    t = int(n*v)
    m = hashlib.sha256()
    m.update(idx.tobytes())
    SHUFFLE_SIG = m.hexdigest()
    # train, valid, test
    return idx[:s], idx[s:(s+t)], idx[(s+t):]


def load_multiclass_csv(data_file, dem=",", target_name=None, sample=None):
    df = pd.read_csv(data_file, delimiter=dem)
    if "name" in df.columns: df = df.drop(columns=["name"])
    if 'smiles' in df.columns:
        df = df.set_index('smiles')
    elif 'SMILES' in df.columns:
        df = df.set_index('SMILES')
    elif 'canonical_smiles' in df.columns:
        df = df.set_index('canonical_smiles')
    else:
        raise RuntimeError("No smile column detected")
        return None
    if target_name:
        clms = [clm for clm in df.columns if clm.startswith(target_name)]
    else:
        clms = [clm for clm in df.columns]
    if len(clms) == 0:
        raise RunTimeError(f"{target_name} not in the dataset")
        return
    clms.sort()
    df = df[clms]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0) # otherwise conflicts with xuefeng's assignment
    df = df.apply(np.abs) # otherwise different from previous results.
    if sample is not None:
        df = df.sample(sample) if isinstance(sample,int) else df.sample(frac=sample)
    return df.index, df.values, df.columns


def try_load_net(model_file=None):
    model_file = Path(model_file)
    if model_file.exists() and model_file.is_file():
        net = torch.load(args.model, map_location=dev)
    else:
        raise FileNotFoundError
    return net.to(dev)


def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    def norm_func(X): return (X-mean) / std
    def restore_func(X): return X * std + mean
    return norm_func, restore_func


def is_valid_smile_for_NFP(sml, max_degree=6):
    """
        NFP requires a valid smile string. 
    """
    try:
        mol = Chem.MolFromSmiles(sml)
        atoms = mol.GetAtoms()
    except:
        warn(f"Not a valid SMILE: {sml}")
        return False

    for atom in atoms:
        if atom.GetDegree() >= max_degree:
            warn(f"larger than max degree {max_degree} {sml}")
            return False
    return True


if __name__ == "__main__":
    """
    This program assumes the canonical smile inputs:
        <three_letter_dataset_short_name>, <molecule_ID_name>, <smiles>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file", help="choose the input csv file",
                        type=str, required=True)
    parser.add_argument("--split_seed", type=int,
                        help="random seed for splitting dataset")
    parser.add_argument("--model", help="choose the pretrained model file for nfp\
                        method.", type=str, required=True)
    parser.add_argument("--target_name", type=str,
                        help="specify the column name")
    parser.add_argument("--tqdm", help="use tqdm progress bar",
                        action="store_true")
    args = parser.parse_args()
    print("#", args)

    INPUT = Path(args.input_file)
    if not INPUT.exists(): raise FileNotFoundError
    SMILES, TARGET, KEYS = load_multiclass_csv(INPUT,
                                               target_name=args.target_name)
    print(f"column names {INPUT.stem} with {len(KEYS)} columns:\
          {KEYS.tolist()}")
    NCLASS = len(KEYS)
    print(f"NCLASS: {NCLASS}")
    net = try_load_net(args.model)
    train_idx, valid_idx, test_idx = \
        split_train_valid_test(len(TARGET), seed=args.split_seed)
    print(f"split_sig: {SHUFFLE_SIG}")
    norm_func, restore_func = normalize_array(
        np.concatenate([TARGET[train_idx], TARGET[valid_idx]], axis=0))
    target = norm_func(TARGET)
    test_data = MolData(SMILES[test_idx], target[test_idx], use_tqdm=args.tqdm)
    test_loader = DataLoader(test_data, batch_size=BSZ, shuffle=False)
    score = net.predict(test_loader)
    gt = TARGET[test_idx]
    prd = restore_func(score)

    res_r2  = []
    res_cor = []
    res_mae = []
    res_mse = []
    if len(prd.shape) == 1: # for single class
        prd = np.expand_dims(prd, 1)
    for idx, k in enumerate(KEYS):
        print(f"idx, k, {idx}, {k}, {prd.shape}, {gt.shape}")
        gt_i, prd_i = gt[:, idx], prd[:, idx]
        res_r2.append(r2_score(gt_i, prd_i))
        res_cor.append(pearsonr(gt_i, prd_i)[0])
        res_mae.append(mae(gt_i, prd_i))
        res_mse.append(mse(gt_i, prd_i))
    
    output_df = pd.DataFrame.from_dict({
        "target": KEYS,
        "r2_score": res_r2,
        "corr_coef": res_cor,
        "mae": res_mae,
        "mse": res_mse})
    output_df.set_index("target")
    table = tabulate(output_df, headers='keys', tablefmt='github',
                     floatfmt=".4f")
    print(table)
    output_df.to_csv('./eval.csv', index=False, float_format="%.4f")

    prd_df = pd.DataFrame.from_dict({k:prd[:,idx] for idx,k in enumerate(KEYS)})
    gt_df = pd.DataFrame.from_dict({k:gt[:,idx] for idx,k in enumerate(KEYS)})
    prd_df.to_csv('./predict.csv', index=False)
    gt_df.to_csv('./ground_truth.csv', index=False)
