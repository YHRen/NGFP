import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from warnings import warn
from tqdm import tqdm

def is_valid_smile_for_NFP(sml, max_degree=6):
    """
        NFP requires a valid smile string. 
    """
    try:
        mol = Chem.MolFromSmiles(sml)
        atoms = mol.GetAtoms()
    except:
        warn(f"Not a valid SMILE: {sml}")
        return 1

    for atom in atoms:
        if atom.GetDegree() >= max_degree:
            warn(f"larger than max degree {max_degree} {sml}")
            return 2
    return 0

def load_multiclass_csv(data_file, dem=",", target_name=None, sample=None):
    df = pd.read_csv(data_file, delimiter=dem)
    if 'smiles' in df.columns:
        df = df.set_index('smiles')
    elif 'SMILES' in df.columns:
        df = df.set_index('SMILES')
    elif 'canonical_smiles' in df.columns:
        df = df.set_index('canonical_smiles')
    else:
        raise RuntimeError("No smile column detected")
        return None
    print(df.columns)
    if "name" in df.columns: df = df.drop(columns=["name"])
    if target_name:
        clms = [clm for clm in df.columns if clm.startswith(target_name)]
        clms.sort()
        if len(clms) == 0:
            raise RunTimeError(f"{target_name} not in the dataset")
            return
        df = df[clms]
    df = df.apply(pd.to_numeric, errors='coerce')
    res_1 = df.isnull().any(1).to_numpy().nonzero()[0]
    res_2 = []
    res_3 = []
    for idx, sml in enumerate(tqdm(df.index)):
        tmp = is_valid_smile_for_NFP(sml) 
        if tmp == 1:#rdkit
            res_2.append(idx)
        elif tmp==2:# > max_degree
            res_3.append(idx)

    print("contains nan:", res_1)
    print("invalid rdkit:", res_2)
    print("larger than 6:", res_3)
    res = set(res_1).union(set(res_2)).union(set(res_3))
    print(len(res_1), len(res_2), len(res_3), len(res))
    return res

#dfile = "./dataset/covid19/xuefeng/docking_data_out_v3.1.csv"
#res = load_multiclass_csv(dfile)
#sr = pd.Series(sorted(list(res)))
#sr.to_csv('./dataset/covid19/xuefeng/invalid_index.csv', index=False,



#dfile = "./dataset/covid19/drug_screening/3CLPro_1_cat_sorted_sample.csv"
dfile = "./dataset/covid19/drug_screening/3CLPro_1_cat_sorted.csv"
res = load_multiclass_csv(dfile, target_name="Chem")
if len(res) > 0:
    sr = pd.Series(sorted(list(res)))
    sr.to_csv('./dataset/covid19/drug_screening/invalid_index.csv', index=False,
          header=["invalid_index"])
