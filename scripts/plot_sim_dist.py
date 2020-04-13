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
import matplotlib.pyplot as plt

from collections.abc import Iterable   # import directly from collections for Python < 3.3

def pd2np(series):
    n, m = len(series), len(series[0])
    res = np.zeros((n, m))
    for i in range(n):
        res[i] = series[i]
    return res

def similarity(df, tgt_idx, column_idx, exclude_zeros=True):
    nfp = pd2np(df['nfp'])
    cfp = pd2np(df['cfp'])
    scr = df.iloc[:,column_idx] if isinstance(column_idx, int) else df[column_idx]
    tgt_nfp, tgt_cfp, tgt_scr = (x[tgt_idx] for x in (nfp, cfp, scr))
    if exclude_zeros:
        nfp, cfp, scr = select_idx(scr != 0, nfp, cfp, scr)
    simnfp = tanimoto_similarity(tgt_nfp, nfp)
    simcfp = tanimoto_similarity(tgt_cfp, cfp)
    difscr = (scr-tgt_scr).abs()
    df2 = pd.DataFrame.from_dict({"nfpsim": simnfp, "cfpsim": simcfp, "difscr":
                                  difscr})
    return df2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="choose the input pkl file",
                        type=str, required=True)
    parser.add_argument("--min", help="pick the smile with largest score\
                        as the anchor smile", action="store_true")
    parser.add_argument("--exclude_zeros", help="exclude smiles with \
                        zero score", action="store_true")
    parser.add_argument("--output_dir", help="output dir for the figures",
                        type=str, default="../figs/")
    args = parser.parse_args()

    df = pd.read_pickle(args.datafile)
    for y_idx in df.columns:
        if "pocket" in y_idx or "Mpro" in y_idx:
            idx = df[y_idx].argmin() if args.min else df[y_idx].argmax()
            df2 = similarity(df, idx, y_idx, exclude_zeros=args.exclude_zeros)
            tmp = df2.hist(bins=50, figsize=(8,4))
            plt.savefig(args.output_dir+str(y_idx)+".png", dpi=300,
                        bbox_inches="tight")
            plt.close()
