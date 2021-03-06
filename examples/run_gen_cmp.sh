#!/bin/bash

datafile=${1:-"../dataset/covid19/6vww_sample10k.csv"}
modelfile=${2:-"../output/6vww_sample20kmulti_class.pkg"}
fplen=${3:-"128"}
topk=${4:-"100"}
interfile=${datafile%.csv}.pkl


echo "generate nfp cfp" $datafile $modelfile $fplen $topk
python generate_nfp_cfp.py --datafile $datafile \
    --model $modelfile \
    --output $interfile \
    --cfp_len $fplen

echo "compare cfp nfp" $interfile
python compare_nfp_cfp.py --datafile $interfile --topk ${topk} --max \
    --demo
