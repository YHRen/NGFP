#!/bin/bash
# python -m torch.utils.bottleneck generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
# python -m cProfile -o cprof.txt generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
#python generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
# python -m cProfile -o cprof.txt generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 252 --num_workers 6 --tqdm --dataset_name DUD >bottle_neck.txt
#python generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --num_workers 6 --tqdm --dataset_name DUD >bottle_neck.txt

exec="generate_nfp_async.py"
for worker in 6 12 24
do
    for bsz in 256 512 1024 2048
    do
        echo "worker" $worker "bsz" $bsz
        python $exec \
            -i ./dataset/canonical_ANL/DUD.csv \
            -o ./output/DUD/ \
            --model ./pretrained/MPro_mergedmulti_class.pkg \
            --chunk_size 10000 \
            --batch_size $bsz  \
            --num_workers $worker \
            --tqdm --dataset_name DUD >bottle_neck.txt
    done
done

# bsz=1000
# worker=6
# python generate_nfp_profile.py  \
#     -i ./dataset/canonical_ANL/DUD.csv \
#     -o ./output/DUD/ \
#     --model ./pretrained/MPro_mergedmulti_class.pkg \
#     --chunk_size 10000 \
#     --batch_size $bsz  \
#     --num_workers $worker \
#     --tqdm --dataset_name DUD >bottle_neck.txt


#python -m cProfile -o cprof.txt generate_nfp_async.py  \
# python generate_nfp_async.py  \
#     -i ./dataset/canonical_ANL/DUD.csv \
#     -o ./output/DUD/ \
#     --model ./pretrained/MPro_mergedmulti_class.pkg \
#     --chunk_size 10000 \
#     --batch_size $bsz  \
#     --num_workers $worker \
#     --tqdm --dataset_name DUD >bottle_neck.txt

