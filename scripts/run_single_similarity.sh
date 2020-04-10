#!/bin/bash

datafile=${1:-"../dataset/covid19/6vww_sample10k.csv"}
interfile=${datafile%.csv}.pkl
modeldir=${2:-"../output/single_task/output/"}
IFS="," read -r -a column_names <<<$(head -n 1 ${datafile}); 
unset IFS
exe1="../examples/generate_nfp_cfp.py"
exe2="../examples/compare_nfp_cfp.py"

N=${#column_names[@]}


rm -f single.md
touch single.md

for (( i=1; i<$N; i++ ))
do
    pckt=${column_names[i]}
    modelfile=${modeldir}ml.${pckt}_dock.pkl.pkg
    echo [[ -e $modelfile ]] 
    python $exe1 --datafile $datafile \
        --model $modelfile \
        --output $interfile 

    python $exe2 --datafile $interfile --max \
    | sed -n -e "/${pckt}/p" >>single.md
done
