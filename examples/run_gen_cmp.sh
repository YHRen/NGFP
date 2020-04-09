#!/bin/bash

datafile=${1:-"../dataset/covid19/6vww_sample10k.csv"}
modelfile=${2:-"../output/6vww_sample20kmulti_class.pkg"}
interfile=${datafile%.csv}.pkl


echo "generate nfp cfp" $datafile $modelfile
python generate_nfp_cfp.py --datafile $datafile \
    --model $modelfile \
    --output $interfile 

echo "compare cfp nfp" $interfile
python compare_nfp_cfp.py --datafile $interfile \
    --demo
