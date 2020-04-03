#!/bin/bash

start_gpu_id=2
end_gpu_id=16 #exclusive
data_dir="../dataset/covid19/"
protein_name="6vww"

total_gpu=$((end_gpu_id - start_gpu_id))
data_files=($(ls ${data_dir}ml.${protein_name}_pocket*_dock.smi))
output_dir="../output/"
sample= 20000

echo ${#data_files[@]}
echo ${data_files[0]}
N=${#data_files[@]}
B=$(( (N+total_gpu)/total_gpu ))

fidx=0

echo N=$N, B=$B, G=$total_gpu

for b in $(seq $B);
do
    for gpu_id in $(seq $start_gpu_id $((end_gpu_id-1)));
    do
        if [ $fidx -ge $N ]
        then
            echo "out" $fidx $N
        else
            echo "gpu_id" $gpu_id fidx $fidx ${data_files[$fidx]} 
            df=${data_files[$fidx]} 
            logf=${df##*/}
            logf=${logf%.smi}.log
            #run code here
            CUDA_VISIBLE_DEVICES=$gpu_id python ../main_covid.py ${df} nfp \
                --delimiter "\t" --output_dir ${output_dir} \
                --sample $sample \
                -b 32 -e 500 -r 5  2>/dev/null >${output_dir}${logf} \
                &
            let "fidx++"
        fi
    done
    wait
done

