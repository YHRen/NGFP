#!/bin/bash
# python -m torch.utils.bottleneck generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
# python -m cProfile -o cprof.txt generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
#python generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
python -m cProfile -o cprof.txt generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 252 --num_workers 6 --tqdm --dataset_name DUD >bottle_neck.txt
#python generate_nfp_profile.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --num_workers 6 --tqdm --dataset_name DUD >bottle_neck.txt
