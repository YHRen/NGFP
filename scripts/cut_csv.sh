#!/bin/bash
dir=${1:-"./"}
# convert csv to tab separated smile format as in ANL data repo.
for i in ${dir}*.csv;
do
    echo $i;
    awk -F, '{ print $5 "\t" $1 "\t" $4}' $i >${i%".csv"}.smi
done
