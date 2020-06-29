# pandas version: 1.05

import pandas as pd
from pathlib import Path

KEY_NAME = "SMILES"
DROP_COL = ["TITLE", "receptor"]
VAL_NAME = "Chemgauss4"
FILE_PTN = "_cat_sorted.csv"
FOLDER   = "/hpcgpfs01/work/csi/covid19/drug_screening/april27"
ROW_LIM  = 700000

def get_name(file_path):
    return str(file_path.name)[:-len(FILE_PTN)]

def process_df(df, name):
    df.drop_duplicates(KEY_NAME, inplace=True)
    df.drop(DROP_COL, axis=1, inplace=True)
    df.rename(columns={VAL_NAME: name}, inplace=True)
    df = df.set_index(KEY_NAME)
    return df

def merge_all_csv():
    folder = Path(FOLDER)
    res = None
    for fp in folder.glob("**/*_cat_sorted.csv"):
        tmp = pd.read_csv(str(fp))
        if len(tmp) < ROW_LIM:
            continue
        name = get_name(fp)
        if res is None:
            res = process_df(tmp, name)
        else:
            tmp = process_df(tmp, name)
            res = pd.concat([res,tmp], axis=1, sort=False, join='outer')
    return res


if __name__ == "__main__":
    df = merge_all_csv()
    df.to_csv('./merged.csv')

