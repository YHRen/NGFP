# Comparing Testing MSE with baseline model

note: baseline model is using traditional fingerprint and a NN model.

### MSE  :arrow_down: 
target             | baseline | NFP    |
|------------------|----------|--------|
|ADRP-ADPR_pocket1 | 0.69     | 0.6180 |
|ADRP-ADPR_pocket5 | 0.31     | 0.2624 |
|ADRP_pocket1      | 0.42     | 0.3608 |
|ADRP_pocket12     | 0.39     | 0.3558 |
|ADRP_pocket13     | 0.57     | 0.5267 |

### R2 Score
| target            | baseline  | NFP    |
|-------------------|-----------|--------|
| ADRP-ADPR_pocket1 | 0.71      | 0.7411 |
| ADRP-ADPR_pocket5 | 0.75      | 0.7932 |
| ADRP_pocket1      | 0.65      | 0.7033 |
| ADRP_pocket12     | 0.63      | 0.6621 |
| ADRP_pocket13     | 0.66      | 0.6920 |

### Pearson Correlation 
| target            | baseline    | NFP    |
|-------------------|-------------|--------|
| ADRP-ADPR_pocket1 | 0.84        | 0.8615 |
| ADRP-ADPR_pocket5 | 0.87        | 0.8917 |
| ADRP_pocket1      | 0.81        | 0.8396 |
| ADRP_pocket12     | 0.8         | 0.8138 |
| ADRP_pocket13     | 0.81        | 0.8338 |

### Concordance Index :arrow_up:

| target            | baseline    | NFP      |
|-------------------|-------------|----------|
| ADRP-ADPR_pocket1 | 0.8229      | 0.8343   |
| ADRP-ADPR_pocket5 | 0.8605      | 0.8774   |
| ADRP_pocket1      | 0.8051      | 0.8096   |
| ADRP_pocket12     | 0.7935      | 0.7925   |
| ADRP_pocket13     | 0.8006      | 0.8048   |


#### For reproducibility

`Namespace(batch_size=32, datafile='../dataset/covid19/xuefeng/ADRP_merged.csv', define_split    =None, delimiter=',', epochs=50, fp_method='nfp', lr=0.001, multiclass=True, output_dir='../ou    tput/', runs=3, sample=None, split_seed=7, target_name='ADRP', use_tqdm=False)`

#### Train-validation-test Split Signature

sha256: `195c6b92b450f6898ca21189ab937061ddbdcf52060131949358ef83d63e80c6`


#### Other evaluation metrics
`Namespace(input_file='../dataset/covid19/xuefeng/ADRP_merged.csv', model='../pretrained/ADRP_mergedmulti_class0.pkg', split_seed=7, tqdm=True)`

`split_sig: 195c6b92b450f6898ca21189ab937061ddbdcf52060131949358ef83d63e80c6`

|    | target            |   r2 score |   corr coef |    mae |    mse |
|----|-------------------|------------|-------------|--------|--------|
|  0 | ADRP-ADPR_pocket1 |     0.7411 |      0.8615 | 0.6148 | 0.6180 |
|  1 | ADRP-ADPR_pocket5 |     0.7932 |      0.8917 | 0.3621 | 0.2624 |
|  2 | ADRP_pocket1      |     0.7033 |      0.8396 | 0.4638 | 0.3608 |
|  3 | ADRP_pocket12     |     0.6621 |      0.8138 | 0.4611 | 0.3558 |
|  4 | ADRP_pocket13     |     0.6920 |      0.8338 | 0.5638 | 0.5267 |


#### Baseline model
(Provided by Xuefeng, older version)

|target            | R2 score | CI   | corr coef | mse    |
|------------------|----------|------|-----------|--------|
|ADRP-ADPR_pocket1 | 0.71     | 0.82 | 0.84      | 0.69   |
|ADRP-ADPR_pocket5 | 0.75     | 0.86 | 0.87      | 0.31   |
|ADRP_pocket1      | 0.65     | 0.8  | 0.81      | 0.42   |
|ADRP_pocket12     | 0.63     | 0.79 | 0.8       | 0.39   |
|ADRP_pocket13     | 0.66     | 0.8  | 0.81      | 0.57   |


(April. 30. newer version)

| target_name              | r2-score    | CI       | correlation | loss       |
|--------------------------|-------------|----------|-------------|------------|
| ADRP-ADPR_pocket1_round1 | 0.7109      | 0.8229   | 0.8432      | 0.6864     |
| ADRP-ADPR_pocket5_round1 | 0.7525      | 0.8605   | 0.8677      | 0.3104     |
| ADRP_pocket1_round1      | 0.6693      | 0.8051   | 0.8183      | 0.4016     |
| ADRP_pocket12_round1     | 0.6433      | 0.7935   | 0.8028      | 0.3720     |
| ADRP_pocket13_round1     | 0.6673      | 0.8006   | 0.8169      | 0.5645     |
