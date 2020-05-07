# PLPro_1 Regression Task
This is trained on the 720k dataset, `plpro_1_cat_sorted.csv` released on April 27th. 


| Target       | Concordance Index | R2 Score     | Pearson Corr. | MAE      | MSE      |
|--------------|-------------------|--------------|---------------|----------|----------|
| PLPro_1      | 0.8356            | 0.7448       | 0.8636        | 0.6910   | 0.8051   |


* MAE: Mean Absolute Error
* MSE: Mean Squared Error

Comparing NFP with a baseline model on regression tasks

### For reproducibility

```
# Namespace(batch_size=32, datafile='../dataset/covid19/drug_screening/plpro_1_cat_sorted.csv', delimiter=',', epochs=50, fp_method='nfp', lr=0.001, output_dir='../output/', runs=3, sample=None, split_seed=7, target_name='Chem', use_tqdm=False)
column names 3CLPro_1_cat_sorted: ['Chemgauss4']
```

```
split_sig: 84674abe718a3323c18455ad20cb01333fa417fb07f51ef25420a63e87c3bdd7
```

<!---
### For comparison

This result is copy-pasted from Xuefeng's baseline model on v3.1 release (300k data).
(Not sure if it is the same pocket. Datasets are labeled differently.)

| Target                | Concordance Index     | R2 Score         | Pearson Corr.     | MSE          |
|-----------------------|-----------------------|------------------|-------------------|--------------|
| PLPro_pocket3_round1(?) | 0.8042                | 0.6596           | 0.8139            | 0.5759
--->
