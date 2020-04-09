import argparse
import numpy as np
import pandas as pd
from pathlib import Path, PurePath
import sys
sys.path.insert(1,str(PurePath(Path.cwd()).parent))
sys.path.insert(1,str(PurePath(Path.cwd())))
from rdkit import DataStructs
from scipy.stats import spearmanr, rankdata
from NeuralGraph.util import dev, tanimoto_similarity
from NeuralGraph.util import tanimoto_similarity
from tabulate import tabulate

from collections.abc import Iterable   # import directly from collections for Python < 3.3


def rdkit_tanimoto(x, y):
    res = []
    if isinstance(y, Iterable):
        for yy in y:
            res.append(DataStructs.TanimotoSimilarity(yy, x))
    else:
        res.append(DataStructs.TanimotoSimilarity(y,x))
    return np.asarray(res)

def pd2np(series):
    n, m = len(series), len(series[0])
    res = np.zeros((n, m))
    for i in range(n):
        res[i] = series[i]
    return res

def get_spearmanr(df, anchor_idx, column_idx):
    idx = anchor_idx
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    simnfp = tanimoto_similarity(nfp[idx], nfp)
    simcfp = tanimoto_similarity(cfp[idx], cfp)
    difscr = (scr-scr[idx]).abs()
    rho_nfp, pval = spearmanr(rankdata(simnfp), rankdata(-difscr))
    rho_cfp, pval = spearmanr(rankdata(simcfp), rankdata(-difscr))
    return rho_nfp, rho_cfp

def get_topk(df, k, anchor_idx, column_idx, mode='overlap'):
    idx = anchor_idx
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    difscr = (scr-scr[idx]).abs()
    simnfp = tanimoto_similarity(nfp[idx], nfp)
    simcfp = tanimoto_similarity(cfp[idx], cfp)
    idx_nfp =  np.argsort(simnfp)[-k:]
    idx_cfp =  np.argsort(simcfp)[-k:]

    if mode=='mean':
        scr_nfp = scr[idx_nfp].mean()
        scr_cfp = scr[idx_cfp].mean()
    elif mode=='overlap':
        scr_idx = set(np.argsort(difscr)[:k])
        scr_nfp = scr_idx.intersection(set(idx_nfp))
        scr_nfp = len(scr_nfp)/k
        scr_cfp = scr_idx.intersection(set(idx_cfp))
        scr_cfp = len(scr_cfp)/k
        
    return scr_nfp, scr_cfp

def demo(df, anchor_idx, column_idx, k=20):
    def get_k(k, srtidx, *argv):
        return (v[srtidx][:k] for v in argv)
    idx = anchor_idx
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    simnfp = tanimoto_similarity(nfp[idx], nfp)
    simcfp = tanimoto_similarity(cfp[idx], cfp)
    difscr = (scr-scr[idx]).abs()
    rho1, pval = spearmanr(rankdata(simnfp), rankdata(-difscr))
    rho2, pval = spearmanr(rankdata(simcfp), rankdata(-difscr))
    n = len(simnfp)
    nfprank, cfprank = n-rankdata(simnfp), n-rankdata(simcfp)
    srtidx = np.argsort(difscr)
    smi, nfpk, cfpk, difk = get_k(k, srtidx,
                                  df.index, simnfp, simcfp, difscr)
    nfprankk, cfprankk = get_k(k, srtidx, nfprank, cfprank)
                                  
    table = tabulate({"smiles": smi, "nfp": nfpk, 
                      "nfp rank": nfprankk,
                      "cfp": cfpk,
                      "cfp rank": cfprankk,
                      "|Δscr|": difk},
                     headers="keys",
                     tablefmt='github')
    print(f"rho NFP: {rho1}, rho CFP: {rho2}")
    print(table)
    return rho1, rho2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="choose the input pkl file",
                        type=str, required=True)
    parser.add_argument("--max", help="pick the smile with largest score\
                        as the anchor smile", action="store_true")
    parser.add_argument("--demo", help="show demo of top 20 closest score", 
                        action="store_true")
    args = parser.parse_args()
    
    K = 100;
    df = pd.read_pickle(args.datafile)
    if args.demo:
        print("DEMO on the last target")
        idx = np.argmax(df.iloc[:,-3])
        demo(df,idx, -3)
        print("=========================\n\n")

    pckt, nfpr, cfpr = [], [], []
    nfpso, cfpso = [], []
    nfpsm, cfpsm = [], []
    for y_idx in df.columns:
        if "pocket" in y_idx:
            idx = df[y_idx].argmax() if args.max else df[y_idx].argmin()
            pckt.append(y_idx)
            r1, r2 = get_spearmanr(df, idx, y_idx)
            nfpr.append(r1)
            cfpr.append(r2)
            s1, s2 = get_topk(df, K, idx, y_idx, mode='overlap')
            nfpso.append(s1)
            cfpso.append(s2)
            s1, s2 = get_topk(df, K, idx, y_idx, mode='mean')
            nfpsm.append(s1)
            cfpsm.append(s2)

    table = tabulate({"pocket": pckt,\
                      "nfp avg": nfpsm, "cfp avg": cfpsm, \
                      "nfp rcl": nfpso, "cfp rcl": cfpso, \
                      "nfp ρ": nfpr, "cfp ρ": cfpr},      \
                     headers="keys",
                     tablefmt='github')
    print(table)
    mean_table = tabulate({"averages":[f"avg top-{K} score mean",\
                                       f"avg top-{K} score recall", "avg spearman corr."],
                           "nfp": [np.asarray(nfpsm).mean(),
                                   np.asarray(nfpso).mean(),
                                   np.asarray(nfpr).mean()],\
                           "cfp": [np.asarray(cfpsm).mean(),
                                   np.asarray(cfpso).mean(),
                                   np.asarray(cfpr).mean()]\
                           }, headers="keys", tablefmt='github')
    print(mean_table)

