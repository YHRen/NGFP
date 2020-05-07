#!/bin/bash
python -m torch.utils.bottleneck generate_nfp.py  -i ./dataset/canonical_ANL/DUD.csv -o ./output/DUD/ --model ./pretrained/MPro_mergedmulti_class.pkg --chunk_size 10000 --batch_size 256 --tqdm --dataset_name DUD >bottle_neck.txt
