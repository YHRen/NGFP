# 3CLPro_1 Regression task

This is trained on the 720k dataset. 

| Target       | Concordance Index | R2 Score     | Pearson Corr. | MAE      | MSE      |
|--------------|-------------------|--------------|---------------|----------|----------|
| 3CLPro_1     | 0.8089            | 0.6756       | 0.8221        | 0.5221   | 0.4593   |


* MAE: Mean Absolute Error
* MSE: Mean Squared Error

Comparing NFP with a baseline model on regression tasks


### For reproducibility

```
# Namespace(batch_size=32, datafile='../dataset/covid19/drug_screening/3CLPro_1_cat_sorted.csv', delimiter=',', epochs=50, fp_method='nfp', lr=0.001, output_dir='../output/', runs=3, sample=None, split_seed=7, target_name='Chem', use_tqdm=False)
column names 3CLPro_1_cat_sorted: ['Chemgauss4']
```

```
split_sig: 3e8638cc224ab0c5d34a55a3daaf013021c6ca9b84232a89c804bf2aa28a3c1a
```

