import re
from pathlib import Path
import pandas as pd
import argparse

def get_pocket_name(s):
    s = Path(s).stem
    s = str(s)
    return s.lstrip("ml.").rstrip("_dock.smi")

def get_number(s):
    z = re.search("\d+", s)
    if z is None: return -1
    return int(z[0])

def merge_smi(args):
    """ merge smi files """
    data_path = Path(args.directory)
    protein_name = args.protein
    df = None
    for fname in data_path.glob("ml."+protein_name+"*_dock.smi"):
        A = pd.read_csv(fname, delimiter='\t')
        A.rename(columns = {'reg': get_pocket_name(fname)}, inplace=True)
        if df is None:
            df = A
        else:
            df = pd.merge(df, A, on=['smiles', 'name'])

    new_columns = df.columns.tolist()
    new_columns.sort(key=get_number)
    df = df[new_columns]
    df = df.set_index("smiles")
    return df

if __name__ == '__main__':
    """ merge all pockets with the same protein name"""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",  type=str,
                        help="Specify the directory name")
    parser.add_argument("protein", type=str,
                        help="Specify the protein name")
    parser.add_argument("-o", "--output", help="output file name")
    parser.add_argument("--format", help="output format",
                        default='csv',
                        choices=['csv', 'pkl'])
    args = parser.parse_args()

    df = merge_smi(args)
    if args.output.endswith("csv"):
        df.to_csv(args.output)
    elif args.output.endswith("pkl"):
        df.to_pickle(args.output)
    elif args.format == "csv":
        df.to_csv(args.output+".csv")
    elif args.format == "pkl":
        df.to_pickle(args.output+".pkl")
    else:
        raise NotImplementedError
