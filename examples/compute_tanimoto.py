import numpy as np
import pandas as pd
import argparse
from pathlib import Path, PurePath
import sys
sys.path.insert(1,str(PurePath(Path.cwd()).parent))
sys.path.insert(1,str(PurePath(Path.cwd())))
from NeuralGraph.util import dev, tanimoto_similarity
from tabulate import tabulate

def pd2np(series):
    series = series.values
    n, m = len(series), len(series[0])
    res = np.zeros((n, m))
    for i in range(n):
        res[i] = series[i]
    return res

def process_data_folder(data_folder):
    input_dir = Path(data_folder)
    assert input_dir.exists() and input_dir.is_dir()
    dnames, mol_names, smls, nfps = [], [], [], []
    files = list(input_dir.glob("**/*.csv"))
    files.sort() # for reproduction purpose
    for csv_file in files:
        with open(csv_file, 'r') as fp:
            for line in fp:
                dname, mol_name, sml, nfp = line.split(',')
                nfp = nfp.split(':')
                nfp = np.asarray([float(x) for x in nfp])
                dnames.append(dname)
                mol_names.append(mol_name)
                smls.append(sml)
                nfps.append(nfp)

    df = pd.DataFrame.from_dict({"dataset": dnames, 
                                 "molecule ID": mol_names,
                                 "smiles": smls,
                                 "nfp": nfps})
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_folder", help="the input csv file\
                        containing neural fingerprints",
                        type=str, required=True)
    parser.add_argument("--anchor_smile_idx", help="the index of the anchor\
                        smile for ranking.", type=int, default=0)
    parser.add_argument("--top_k", help="return the top k results.",
                        default=20, type=int)
    args = parser.parse_args()

    tpk, ank = args.top_k, args.anchor_smile_idx
    
    ## load smile strings and fingerprints
    df = process_data_folder(args.data_folder)

    ## compute similarities to the anchor smile of the fingerprints 
    nfp = pd2np(df['nfp'])
    ank = args.anchor_smile_idx
    similarities = tanimoto_similarity(nfp[ank], nfp)

    ## find top k, excluding itself
    sorted_idx = np.argsort(-similarities)
    top_idx = sorted_idx[:(tpk+1)] # top k+1
    top_idx = top_idx[top_idx!=ank][:tpk] # exclude itself
    top_sml = [df['smiles'][i] for i in top_idx]
    top_score = similarities.take(top_idx)

    ## show results
    print(f"among total {len(df)} molecules")
    print(f"top-{tpk} similar smiles to {df['smiles'][ank]}")
    table = tabulate(zip(top_sml, top_score),
                     headers = ["smiles", "score"], tablefmt='github')
    print(table)
