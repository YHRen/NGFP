#!/bin/bash

RUN=3
# 6vww_sample10k.csv
data_file=$1
IFS="," read -r -a column_names <<<$(head -n 1 ${data_file}); 
unset IFS
start_gpu_id=13
end_gpu_id=16 #exclusive
protein_name="6vww"
total_gpu=$((end_gpu_id - start_gpu_id))
output_dir="../output/"
N=${#column_names[@]}
B=$(( (N+total_gpu)/total_gpu ))
idx=1
echo N=$N, B=$B, G=$total_gpu
for b in $(seq $B);
do
    for gpu_id in $(seq $start_gpu_id $((end_gpu_id-1)));
    do
        if [ $idx -ge $N ]
        then
            echo "out" $idx $N
        else
            echo "gpu_id" $gpu_id idx $idx ${column_names[$idx]} 
            df=${column_names[$idx]} 
            logf=${data_file##*/}
            logf=${logf%.csv}.log
            #run code here
            CUDA_VISIBLE_DEVICES=$gpu_id python ../main_covid.py ${df} nfp \
                --output_dir ${output_dir} \
                --target_name $df \
                --split_seed 7 \
                -b 32 -e 50 -r $RUN  2>/dev/null >${output_dir}${logf} \
                &
            let "idx++"
        fi
    done
    wait
done

