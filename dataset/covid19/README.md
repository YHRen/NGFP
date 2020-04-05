CSV data are downloaded from https://app.box.com/folder/106580580881
and processed with the following bash script. 


    * extract relevant columns from the csv files and 
    * rearrange the columns such that the first column is smile strings
    * convert csv to tab separated format 
    * the outcome is aligned with other datasets in ANL data repo.
    * https://argonne-data-repo.readthedocs.io/en/latest/

```bash
#!/bin/bash
for i in *_dock.csv;
do
    echo $i;
    awk -F, '{ print $5 "\t" $1 "\t" $4}' $i >${i%".csv"}.smi
done
```


## load csv file 
```
df = pd.read_csv('6vww_sample10k.csv', index_col=0)
```
