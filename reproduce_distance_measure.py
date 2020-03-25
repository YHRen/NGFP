from torch.utils.data import DataLoader, Subset
from pathlib import Path
import torch
import torch.nn as nn
import itertools as its
import pandas as pd
import numpy as np
import json
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

from NeuralGraph.dataset import MolData
from NeuralGraph.model import QSAR
from NeuralGraph.util import dev

def tanimoto_distance(x, y):
    idx = x<=y
    return 1 - (x[idx].sum() + y[~idx].sum()) / (x[~idx].sum() + y[idx].sum())

def get_circular_fp(smile, radius=6, fp_len=128):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius, fp_len)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

def get_neural_fp(X, net):
    x0, x1, x2 = X
    x0, x1, x2 = x0.to(dev), x1.to(dev), x2.to(dev)
    x0, x1, x2 = (torch.unsqueeze(x, 0) for x in (x0, x1, x2))
    res = net.nfp(x0, x1, x2)
    res = res.detach().cpu().numpy()
    return res

def mse(x, y):
    return ((x-y)**2).mean()

def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    A_normed = (A - mean) / std
    def restore_function(X):
        return X * std + mean
    return A_normed, restore_function

def change_net_to_weights(net, lo_bnd, hi_bnd):
    for n,m in net.named_children():
        if isinstance(m, torch.nn.Linear):
            nn.init.uniform_(m.weight, lo_bnd, hi_bnd)
            if m.bias is not None:
                nn.init.uniform_(m.bias, lo_bnd, hi_bnd)
        change_net_to_weights(m, lo_bnd, hi_bnd)

def calc_distance(net, data, smiles, FP_LEN,\
                    sample_sz=1000, SEED=None):
    N, sample_sz = len(data), sample_sz
    if SEED: np.random.seed(SEED)
    res = [[],[]]
    for _ in range(sample_sz):
        i, j = np.random.choice(N, 2)
        dst0 = tanimoto_distance(get_circular_fp(smiles[i], fp_len=FP_LEN),
                                 get_circular_fp(smiles[j], fp_len=FP_LEN))
        dst1 = tanimoto_distance(get_neural_fp(data[i][0], net),
                                 get_neural_fp(data[j][0], net))
        res[0].append(dst0)
        res[1].append(dst1)

    res = np.asarray(res)
    return res

def calc_corr(res):
    return (np.corrcoef(res[0], res[1])[0,1])

def plot_scatter(net, data, smiles, FP_LEN, filename,\
                 sample_sz = 1000, SEED=None):
    res = calc_distance(net, data, smiles, FP_LEN, \
                          sample_sz, SEED)
    plt.scatter(res[0], res[1], marker='o', facecolors='none', edgecolors='b', alpha=0.3)
    plt.xlabel("circular fingerprint distance")
    plt.ylabel("neural fingerprint distance")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Correlation = {:.4f}".format(np.corrcoef(res[0], res[1])[0,1]))
    plt.savefig(filename, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # Load Data
    DATAFILE = Path('./dataset/solubility/delaney-processed.csv')
    df = pd.read_csv(DATAFILE)
    target = df['measured log solubility in mols per litre'].values
    target, restore = normalize_array(target)
    data = MolData(df['smiles'], target)
    print(type(df['smiles'][0]), df['smiles'][0])
    tmp = df['smiles'][0]
    print(get_circular_fp(tmp))
    exit()

    # Plot with a random weight and 2048 length as in Figure3Left
    gcn_act = ['sigmoid', 'relu', 'tanh']
    gop_act = ['sigmoid', 'tanh', 'softmax']
    large_weights = [(-1e7, 1e7), (0, 1e7), (-1e3, 1e3), (-10, 10)]
    max_degs = [1, 6]
    res = {}
    for a1, a2, bnds, rd in its.product(gcn_act, gop_act, large_weights,
                                        max_degs):
        SEED, FP_LEN = 7, 1<<11 
        net = QSAR(hid_dim=FP_LEN, n_class=1, max_degree=rd,
                   gcn_activation=a1,
                   gop_activation=a2)
        print("nbnds", bnds)
        change_net_to_weights(net.nfp, *bnds)
        tmp = calc_distance(net, data, df['smiles'], FP_LEN, sample_sz=500,
                            SEED=7)
        tmp = calc_corr(tmp)
        res[f"gcn-{a1}_gop-{a2}_weights-{bnds}_radius-{rd}"]=tmp
        print(f"gcn-{a1}_gop-{a2}_weights-{bnds}_radius-{rd}", tmp)


    with open('./output.json', 'w') as fp:
        json.dump(res, fp)


    exit()
    plot_scatter(net,
                 data,
                 df['smiles'],
                 FP_LEN,
                 "./figs/scatter_nfp_vs_cfp_2048_random_weight.png")

    exit()
    # Plot with a trained model
    OUTPUT = './output/best_delaney.pkl'
    net = torch.load(OUTPUT+'.pkg')
    SEED, FP_LEN = 7, 1<<11 
    plot_scatter(net,
                 data,
                 df['smiles'],
                 FP_LEN,
                 "./figs/scatter_nfp_vs_cfp_128_trained_weight.png")

