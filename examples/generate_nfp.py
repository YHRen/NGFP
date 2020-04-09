import torch
import argparse
import numpy as np
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

def line_parser(line, dmt=None, idx=0):
    """ parse the line and get the smile string.  """
    return line.split(dmt)[idx]

def oscillator(period):
    x, y = 1, period
    def f():
        nonlocal x
        z = x==0
        x = (x+1)%y
        return z
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="choose the input smi file",
                        type=str, required=True)
    parser.add_argument("-d", "--delimiter", help="choose the delimiter of the smi\
                        file", type=str, default=" ")
    parser.add_argument("-i", "--column_index", help="choose the column_index of\
                        the smile strings.", type=int, default=0)
    parser.add_argument("--skip", help="choose the number of lines to skip\
                        .", type=int, default=0)
    mux_group = parser.add_mutually_exclusive_group(required=True)
    mux_group.add_argument("--model", help="choose the saved model file for nfp\
                        method. If not specified, large random weights would\
                        be used", type=str)
    mux_group.add_argument("--morgan", help="use circular fingerprint method \
                        (Morgan)", type=int)
    parser.add_argument("--output", help="specify the output file (npy)",
                        default="./output/example_nfp_output.npy")
    args = parser.parse_args()

    if args.model:
        net = try_load_net(args.model)
        dmt, idx = args.delimiter, args.column_index
        res = []
        bsz = 1<<11
        with open(args.datafile,'r') as fp:
            osc, cache = oscillator(bsz), []
            if args.skip > 0:
                for _ in zip(fp, range(args.skip)): continue
            for line in tqdm(fp):
                if osc():
                    res.append(net.calc_nfp(cache))
                    cache = []
                sml = line_parser(line, dmt=dmt, idx=idx)
                cache.append(sml)

            if len(cache) > 0:
               res.append(net.calc_nfp(cache))
    else: # args.morgan
        dmt, idx = args.delimiter, args.column_index
        fp_len = args.morgan if args.morgan > 0 else 128
        res = []
        with open(args.datafile,'r') as fp:
            if args.skip > 0:
                for _ in zip(fp, range(args.skip)): continue
            for line in tqdm(fp):
                sml = line_parser(line, dmt=dmt, idx=idx)
                res.append(get_circular_fp(sml, fp_len=args.morgan))

    res = np.concatenate(res)
    np.save(args.output, res)
