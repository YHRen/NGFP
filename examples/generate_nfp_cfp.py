import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path, PurePath
import sys
sys.path.insert(1,str(PurePath(Path.cwd()).parent))
sys.path.insert(1,str(PurePath(Path.cwd())))
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from NeuralGraph.model import QSAR
from NeuralGraph.util import dev, enlarge_weights

def get_circular_fp(smile, radius=6, fp_len=128):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius, fp_len)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

def try_load_net(model_file=None):
    if model_file is not None:
        model_file = Path(model_file)
        if model_file.exists() and model_file.is_file():
            net = torch.load(args.model, map_location=dev)
        else:
            raise FileNotFoundError
    else: # random large weights
        net = QSAR(hid_dim=128, n_class=1, max_degree=6)
        enlarge_weights(net, -1e4, 1e4)
    return net.to(dev)

def compute_nfp_cfp(args):
    df = pd.read_csv(args.datafile);
    keys = df['smiles'].tolist()
    nfp, cfp = [], []
    bsz = 1<<9
    net = try_load_net(args.model)
    for idx in range(0, len(keys), bsz):
        cache = keys[idx:idx+bsz]
        nfp.extend(net.calc_nfp(cache))

    for smi in keys:
        cfp.append(get_circular_fp(smi))

    df['nfp'] = nfp
    df['cfp'] = cfp
    df = df.set_index("smiles")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="choose the input smi file",
                        type=str, required=True)
    parser.add_argument("--model", help="choose the saved model file for nfp\
                        method. If not specified, large random weights would\
                        be used", type=str, required=True)
    parser.add_argument("--output", help="specify the output file pandas\
                        pickle file",
                        default="../output/example_nfp_cfp.pkl")
    args = parser.parse_args()
    df = compute_nfp_cfp(args)
    df.to_pickle(args.output)
