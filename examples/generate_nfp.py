import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path, PurePath
import sys
sys.path.insert(1,str(PurePath(Path.cwd()).parent))
sys.path.insert(1,str(PurePath(Path.cwd())))
from NeuralGraph.model import QSAR
from NeuralGraph.util import dev, enlarge_weights

def try_load_net(model_file=None):
    if model_file is not None:
        model_file = Path(model_file)
        if model_file.exists() and model_file.is_file():
            net = torch.load(args.model)
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
    parser.add_argument("--delimiter", help="choose the delimiter of the smi\
                        file", type=str, default=" ")
    parser.add_argument("--column_index", help="choose the column_index of\
                        the smile strings.e", type=int, default=0)
    parser.add_argument("--model", help="choose the saved model file for nfp\
                        method. If not specified, large random weights would\
                        be used", type=str, required=True)
    parser.add_argument("--output", help="specify the output file (npy)",
                        default="./output/example_nfp_output.npy")
    args = parser.parse_args()

    net = try_load_net(args.model)
    dmt, idx = args.delimiter, args.column_index
    res = []
    bsz = 1<<12
    with open(args.datafile,'r') as fp:
        osc, cache = oscillator(bsz), []
        for line in tqdm(fp):
            if osc():
                res.append(net.calc_nfp(cache))
                cache = []
            else:
                sml = line_parser(line, dmt=dmt, idx=idx)
                cache.append(sml)

        if len(cache) > 0:
           res.append(net.calc_nfp(cache))


    res = np.concatenate(res)
    res = np.float(res)
    np.save(args.output, res)
