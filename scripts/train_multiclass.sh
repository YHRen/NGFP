#!/bin/bash
RUN=5
data_file=$1
gpu_id=${2:-0}
output_dir="../output/"
df="multiclass"
logf=${data_file##*/}
logf=${logf%.csv}${df}.log
echo $data_file $logf

CUDA_VISIBLE_DEVICES=$gpu_id \
python ../main_covid.py ${data_file} nfp \
       --output_dir ${output_dir} \
       --multiclass  \
       --split_seed 7 \
       -b 32 -e 50 -r $RUN \
       2>${output_dir}${logf%log}err \
       >${output_dir}${logf}
