from pathlib import Path
import torch
import argparse

from NeuralGraph.preprocessing import tensorise_smiles
from NeuralGraph.model import QSAR
from NeuralGraph.util import (
    dev,
    calc_circular_fp,
    calc_neural_fp,
    tanimoto_distance,
    enlarge_weights
)

FP_LEN = 1<<7 # 1<<11
METHODS = ["morgan", "nfp"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("s1", default="CN=C=O", type=str,
                        help="the first SMILE string")
    parser.add_argument("s2", default="O=Cc1ccc(O)c(OC)c1COCc1cc(C=O)ccc1O",
                        type=str,
                        help="the second SMILE string")
    parser.add_argument("-m", "--method",
                        help="choose the fingerprint method to compute\
                        similarity score",
                        default="nfp", choices=METHODS)
    parser.add_argument("--model", help="choose the saved model file for nfp\
                        method. If not specified, large random weights would\
                        be used", type=str)

    args = parser.parse_args()
    if args.method == METHODS[0]:
        fp1 = calc_circular_fp(args.s1, radius=6, fp_len=FP_LEN)
        fp2 = calc_circular_fp(args.s2, radius=6, fp_len=FP_LEN)
        print(1-tanimoto_distance(fp1, fp2))
    elif args.method == METHODS[1]:
        if args.model is not None:
            model_file = Path(args.model)
            if model_file.exists() and model_file.is_file():
                net = torch.load(args.model)
            else:
                raise FileNotFoundError
        else:
            net = QSAR(hid_dim=FP_LEN, n_class=1, max_degree=6)
            enlarge_weights(net, -1e4, 1e4)
        
        tmp = tensorise_smiles([args.s1, args.s2])
        fp1, fp2 = calc_neural_fp(tmp, net)
        print(1-tanimoto_distance(fp1, fp2))
