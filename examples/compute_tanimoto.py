import numpy as np
import pandas as pd
import argparse
from pathlib import Path, PurePath
from tabulate import tabulate
try:
    import NeuralGraph
except:
    import sys
    sys.path.insert(1,str(PurePath(Path.cwd()).parent))
    sys.path.insert(1,str(PurePath(Path.cwd())))
from NeuralGraph.util import dev, tanimoto_similarity, is_valid_smile
from NeuralGraph.nfp import nfp_net

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


def compute_nfp(net, sml):
    if not is_valid_smile(sml):
        raise Exception(f"{sml} is invaild")
    return net.calc_nfp([sml])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_folder", help="the input csv file\
                        containing neural fingerprints",
                        type=str, required=True)
    parser.add_argument("-q", "--query", help="input smile string",
                        type=str)
    parser.add_argument("-k", "--top_k", help="return the top k results.",
                        default=20, type=int)
    args = parser.parse_args()

    tpk = args.top_k
    if not args.query:
        args.query = "CCCCCC1=CC(O)=C(C\C=C(/C)CCC=C(C)C)C(O)=C1"
    
    ## load smile strings and fingerprints
    df = process_data_folder(args.data_folder)
    nfp = pd2np(df['nfp'])

    ## load net remotely
    net = nfp_net(pretrained=True, protein="Mpro", progress=True)

    ## compute similarities to the anchor smile of the fingerprints 
    rst_nfp = compute_nfp(net, args.query)
    similarities = tanimoto_similarity(rst_nfp, nfp)

    ## find top k, excluding itself
    sorted_idx = np.argsort(-similarities)
    top_idx = sorted_idx[:tpk] # top k+1
    top_sml = [df['smiles'][i] for i in top_idx]
    top_score = similarities.take(top_idx)

    ## show results
    print(f"among total {len(df)} molecules")
    print(f"top-{tpk} similar smiles to {args.query}")
    table = tabulate(zip(top_sml, top_score),
                     headers = ["smiles", "score"], tablefmt='github')
    print(table)
