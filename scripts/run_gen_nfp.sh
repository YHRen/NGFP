#!/bin/bash
cd "$(dirname "$0")"
DATASET=${1:-DUD_sample.csv}
CHUNK_SZ=${2:-100}
OUTPUT=${DATASET%.csv}
echo $DATASET $OUTPUT $CHUNK_SZ
python ../generate_nfp.py \
    -i ../dataset/canonical_ANL/${DATASET}`# input file`\
    -o ../output/${OUTPUT}/ `# output directory`\
    --model ../pretrained/MPro_mergedmulti_class.pkg \
    --chunk_size ${CHUNK_SZ} `# for demo purpose`\
    --batch_size 32 \
    --tqdm \
    --dataset_name ${OUTPUT} # if not defined, will derive from input
    #--model ./pretrained/MPro_mergedmulti_class.pkg \
    #--local_model ../pretrained/MPro_mergedmulti_class.pkg \
