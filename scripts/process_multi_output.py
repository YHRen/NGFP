import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

def parse_multiclass_log(filename):
    keys = None
    vals = []
    with open(filename, 'r') as fp:
        for line in fp:
            if line.startswith("column names"):
                _, tmp = line.split(":")
                keys = tmp.strip().lstrip('[').rstrip(']').split(",")
                keys = [k.strip().strip("'") for k in keys]
            elif line.startswith("mse_percls"):
                for _ in range(3):
                    line+=fp.readline()
                _, tmp = line.split(":")
                tmp = tmp.strip().lstrip('[').rstrip(']')
                tmp = tmp.replace('\n',' ')
                tmp = tmp.split()
                tmp = [float(x) for x in tmp]
                vals.append(tmp)

    vals = np.asarray(vals)  # runs by pockets
    return keys, vals
                


data_dir = "../output/"
protein_name = "MPro"
#filename = data_dir+protein_name+"_sample"+sz+"multiclass.log"
filename = data_dir+protein_name+"_mergedmulticlass.log"
keys, vals = parse_multiclass_log(filename)
df = pd.DataFrame.from_dict({"keys": keys, "MSE": vals.mean(axis=0), "MSE std":
                             vals.std(axis=0)})
df = df.set_index("keys")
print(tabulate(df, headers=df.columns))

orgcsv = "../dataset/covid19/MPro_merged.csv"
df2 = pd.read_csv(orgcsv)
df2 = df2.set_index("smiles")
df2 = df2.drop(columns=["name"])
df2desc = df2.describe()
df2desc = df2desc.transpose()
df2desc.to_csv("../output/MPro_describe.csv")
df2desc.index = df2desc.index.rename("keys")

df3 = df.merge(df2desc, on="keys")
df3["normalized MSE"] = df3["MSE"]/df3["std"]**2
print(tabulate(df3, headers=df3.columns, tablefmt="github"))
