import numpy as np
import argparse
from pathlib import Path, PurePath
import sys
sys.path.insert(1,str(PurePath(Path.cwd()).parent))
sys.path.insert(1,str(PurePath(Path.cwd())))
from NeuralGraph.util import dev, tanimoto_similarity
from tabulate import tabulate

def example():
    """ demostrate the continuous tanimoto similarity

    x: a neural fingerprint of length 128
    y: a neural fingerprint of length 128
    z: 8 neural fingerprints of length 128. (8x128 numpy matrix)
    """
    x = np.random.rand(128)
    y = np.random.rand(128)
    z = np.random.rand(4, 128)
    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")
    print(f"z.shape = {z.shape}")
    print("tanimoto(x,x) =", tanimoto_similarity(x,x))
    print("tanimoto(x,y) =", tanimoto_similarity(x,y))
    print("tanimoto(x,z) =", tanimoto_similarity(x,z))



def line_parser(line, dmt=None, idx=0):
    """ parse the line and get the smile string.  """
    return line.split(dmt)[idx]

def load_smiles(filename, dmt=None, idx=0):
    res = []
    with open(filename, 'r') as fp:
        for line in fp:
            res.append(line_parser(line, dmt, idx))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="the input smi file",
                        type=str, required=True)
    parser.add_argument("--fingerprint", help="the generated fingerprint \
                        file", type=str, required=True)
    parser.add_argument("--anchor_smile_idx", help="the index of the anchor\
                        smile for ranking.", type=int, default=0)
    parser.add_argument("--top_k", help="return the top k results.",
                        default=10, type=int)
    parser.add_argument("--delimiter", help="choose the delimiter of the smi\
                        file", type=str, default=" ")
    parser.add_argument("--column_index", help="choose the column_index of\
                        the smile strings.", type=int, default=0)
    args = parser.parse_args()

    dmt, idx = args.delimiter, args.column_index
    tpk, ank = args.top_k, args.anchor_smile_idx
    
    ## load smile strings and fingerprints
    nfp = np.load(args.fingerprint)
    sml = load_smiles(args.datafile)

    ## compute similarities to the anchor smile of the fingerprints 
    similarities = tanimoto_similarity(nfp[ank], nfp)

    ## find top k, excluding itself
    sorted_idx = np.argsort(-similarities)
    top_idx = sorted_idx[:(tpk+1)] # top k+1
    top_idx = top_idx[top_idx!=ank][:tpk] # exclude itself
    top_sml = [sml[i] for i in top_idx]
    top_score = similarities.take(top_idx)

    ## show results
    print(f"top-{tpk} similar smiles to {sml[ank]}")
    table = tabulate(zip(top_sml, top_score),
                     headers = ["smiles", "score"])
    print(table)
