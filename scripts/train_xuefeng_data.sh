#!/bin/bash
RUN=5
xuefeng_data_dir="../dataset/covid19/xuefeng/"
data_file=${1:-${xuefeng_data_dir}"docking_data_out_v3.1.csv"}
train_idx=${2:-${xuefeng_data_dir}"train_row_index_suggested.csv"}
valid_idx=${3:-${xuefeng_data_dir}"val_row_index_suggested.csv"}
test_idx=${4:-${xuefeng_data_dir}"test_row_index_suggested.csv"}
protein_name=${5:-"ADRP"}
df="xuefeng"
output_dir="../output/"
logf=${data_file##*/}
logf=${logf%.csv}${df}.log
echo $data_file $logf
python ../main_covid_xuefeng.py ${data_file} nfp \
       --output_dir ${output_dir} \
       --multiclass  \
       --target_name ${protein_name} \
       --define_split ${train_idx} ${valid_idx} ${test_idx} \
       -b 32 -e 50 -r $RUN \
       2>${output_dir}${logf%log}err \
       >${output_dir}${logf}
