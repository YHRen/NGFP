import pandas as pd
from lifelines.utils import concordance_index


prd = pd.read_csv('./predict.csv', dtype=float)
gt = pd.read_csv('./ground_truth.csv', dtype=float)

for c1, c2 in zip(prd.columns, gt.columns):
    assert c1 == c2

for k in prd.columns:
    print(k, ":", concordance_index(gt[k], prd[k]))
