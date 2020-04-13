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

def select_idx(zero_idx, *argv):
    return (x[zero_idx] for x in argv)

def get_spearmanr(df, tgt_idx, column_idx, exclude_zeros=True):
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    tgt_nfp, tgt_cfp, tgt_scr = (x[tgt_idx] for x in (nfp, cfp, scr))

    if exclude_zeros:
        nfp, cfp, scr = select_idx(scr != 0, nfp, cfp, scr)
    
    simnfp = tanimoto_similarity(tgt_nfp, nfp)
    simcfp = tanimoto_similarity(tgt_cfp, cfp)
    difscr = (scr-tgt_scr).abs()
    rho_nfp, pval = spearmanr(rankdata(simnfp), rankdata(-difscr))
    rho_cfp, pval = spearmanr(rankdata(simcfp), rankdata(-difscr))
    return rho_nfp, rho_cfp

def get_topk(df, k, tgt_idx, column_idx, mode='overlap', exclude_zeros=True):
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    tgt_nfp, tgt_cfp, tgt_scr = (x[tgt_idx] for x in (nfp, cfp, scr))

    if exclude_zeros:
        nfp, cfp, scr = select_idx(scr!=0, nfp, cfp, scr)

    simnfp = tanimoto_similarity(tgt_nfp, nfp)
    simcfp = tanimoto_similarity(tgt_cfp, cfp)
    difscr = (scr-tgt_scr).abs()
    idx_nfp =  np.argsort(simnfp)[-k:]
    idx_cfp =  np.argsort(simcfp)[-k:]

    if mode=='mean':
        scr_nfp = scr[idx_nfp].mean()
        scr_cfp = scr[idx_cfp].mean()
    elif mode=='overlap':
        scr_idx = set(np.argsort(difscr)[:k]) # top closest smiles
        scr_nfp = scr_idx.intersection(set(idx_nfp))
        scr_nfp = len(scr_nfp)/k
        scr_cfp = scr_idx.intersection(set(idx_cfp))
        scr_cfp = len(scr_cfp)/k
        
    return scr_nfp, scr_cfp

def demo(df, tgt_idx, column_idx, k=20, exclude_zeros=True, sorted_by="nfp"):
    """
    sorted_by : nfp, cfp, scr
    """
    def get_k(k, srtidx, *argv):
        return (v[srtidx][:k] for v in argv)
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    tgt_nfp, tgt_cfp, tgt_scr = (x[tgt_idx] for x in (nfp, cfp, scr))
    if exclude_zeros:
        nfp, cfp, scr = select_idx(scr != 0, nfp, cfp, scr)
    simnfp = tanimoto_similarity(tgt_nfp, nfp)
    simcfp = tanimoto_similarity(tgt_cfp, cfp)
    difscr = (scr-tgt_scr).abs()
    rho1, pval = spearmanr(rankdata(simnfp), rankdata(-difscr))
    rho2, pval = spearmanr(rankdata(simcfp), rankdata(-difscr))
    n = len(simnfp)
    nfprank, cfprank = n-rankdata(simnfp), n-rankdata(simcfp)
    if sorted_by == "nfp":
        srtidx = np.argsort(nfprank)
    elif sorted_by == "cfp":
        srtidx = np.argsort(cfprank)
    elif sorted_by == "scr":
        srtidx = np.argsort(difscr) # sort by ground truth
    else:
        raise NotImplementedError
    
    smi, nfpk, cfpk, scrk, difk = get_k(k, srtidx,
                                  df.index, simnfp, simcfp, scr, difscr)
    nfprankk, cfprankk = get_k(k, srtidx, nfprank, cfprank)
                                  
    table = tabulate({"smiles": smi, "nfp": nfpk, 
                      "nfp rank": nfprankk.astype('int32'),
                      "cfp": cfpk,
                      "cfp rank": cfprankk.astype('int32'),
                      "scr" : scrk,
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
    parser.add_argument("--exclude_zeros", help="exclude smiles with \
                        zero score", action="store_true")
    parser.add_argument("--demo", help="show demo of top 20 closest score", 
                        action="store_true")
    parser.add_argument("--topk", help="show demo of top K closest score", 
                        type=int, default=100)
    args = parser.parse_args()
    
    K = args.topk
    df = pd.read_pickle(args.datafile)
    if args.demo:
        print(f"DEMO on the last target: {df.columns[-3]}")
        idx = np.argmax(df.iloc[:,-3])
        demo(df,idx, -3)

    pckt, nfpr, cfpr = [], [], []
    nfpso, cfpso = [], []
    nfpsm, cfpsm = [], []
    for y_idx in df.columns:
        if "pocket" in y_idx or "Mpro" in y_idx:
            idx = df[y_idx].argmax() if args.max else df[y_idx].argmin()
            pckt.append(y_idx)
            r1, r2 = get_spearmanr(df, idx, y_idx,
                                   exclude_zeros=args.exclude_zeros)
            nfpr.append(r1)
            cfpr.append(r2)
            s1, s2 = get_topk(df, K, idx, y_idx, mode='overlap',
                                   exclude_zeros=args.exclude_zeros)
            nfpso.append(s1)
            cfpso.append(s2)
            s1, s2 = get_topk(df, K, idx, y_idx, mode='mean',
                                   exclude_zeros=args.exclude_zeros)
            nfpsm.append(s1)
            cfpsm.append(s2)

    table = tabulate({"pocket": pckt,\
                      "nfp avg": nfpsm, "cfp avg": cfpsm, \
                      "nfp rcl": nfpso, "cfp rcl": cfpso, \
                      "nfp ρ": nfpr, "cfp ρ": cfpr},      \
                     headers="keys",
                     tablefmt='github',
                     floatfmt='.4f')
    print(table)
    mean_table = tabulate({"averages":[f"avg top-{K} score mean",\
                                       f"avg top-{K} score recall", "avg spearman corr."],
                           "nfp": [np.asarray(nfpsm).mean(),
                                   np.asarray(nfpso).mean(),
                                   np.asarray(nfpr).mean()],\
                           "cfp": [np.asarray(cfpsm).mean(),
                                   np.asarray(cfpso).mean(),
                                   np.asarray(cfpr).mean()]\
                           }, headers="keys", tablefmt='github',
                          floatfmt='.4f')
    print(mean_table)

