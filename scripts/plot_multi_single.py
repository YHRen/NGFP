import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_single_task_log(filename, protein):
    with open(filename, 'r') as fp:
        for line in fp:
            if line.startswith(protein):
                line=line.strip()
                _, avg, std = re.split(": |,", line)
                return avg, std
    return None, None

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
                

subset_names = ["5k", "10k", "20k"]
protein_name = "6vww"

#test_file = "../output/6vww_sample10k6vww_pocket18.log"
#test_file2 = "../output/6vww_sample10kmulticlass.log"
#print(parse_single_task_log(test_file, "6vww"))
#print(parse_multiclass_log(test_file2))

data_dir = "../output/"

def get_glob(protein_name, samplesz):
    return protein_name+"_sample"+samplesz+"*_pocket*.log"

def get_single_results(protein_name, sample_sz):
    global data_dir
    keys = []
    avgs, stds = [], []
    pattern = get_glob(protein_name, sample_sz)
    for filename in Path(data_dir).glob(pattern):
        avg, std = parse_single_task_log(str(filename), protein_name)
        key = filename.stem
        idx = key.find(protein_name+"_pocket")
        key = key[idx:].strip("'")
        keys.append(key)
        avgs.append(float(avg))
        stds.append(float(std))
    return keys, np.asarray(avgs), np.asarray(stds)

#  get single task
df, ds = None, None
for sz in subset_names:
    keys, avgs, stds = get_single_results(protein_name, sz)
    colname = "single"+sz
    tmp = {"keys": keys, colname: avgs}
    if df is None:
        df = pd.DataFrame.from_dict(tmp)
        df = df.set_index("keys")
        ds = pd.DataFrame.from_dict({"keys": keys, colname:
                                     stds}).set_index("keys")
    else:
        tmp = pd.DataFrame.from_dict(tmp)
        tmp = tmp.set_index("keys")
        tmq = pd.DataFrame.from_dict({"keys": keys, colname: stds})
        tmq = tmq.set_index("keys")
        df = df.join(tmp)
        ds = ds.join(tmq)

# get multiclass
for sz in subset_names:
    filename = data_dir+protein_name+"_sample"+sz+"multiclass.log"
    keys, vals = parse_multiclass_log(filename)
    colname = "multi"+sz
    tmp = {"keys": keys, colname: vals.mean(axis=0)}
    tmq = {"keys": keys, colname: vals.std(axis=0)}
    df = df.join(pd.DataFrame.from_dict(tmp).set_index("keys"))
    ds = ds.join(pd.DataFrame.from_dict(tmq).set_index("keys"))

df = df.sort_values(by=["single20k"])
ds = ds.reindex(df.index)
print(df.mean(axis=0))
x = list(range(df.shape[0]))
for k in df.columns:
    plt.errorbar(x, df[k], yerr=ds[k], label=k)
plt.legend()
tp = [str(x[len(protein_name)+1:]) for x in df.index]
tp = [x.replace("pocket", "pckt") for x in tp]
plt.title(protein_name)
plt.ylabel("mse")
plt.xticks(x, tp, rotation=45, ha='right')
plt.savefig("./test.png", dpi=300, bbox_inches="tight")
