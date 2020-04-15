#!/bin/bash
DATASET=${1:-DUD_sample.csv}
OUTPUT=${2:-${DATASET%.csv}}
CHUNK_SZ=${3:-5000}
echo $DATASET $OUTPUT $CHUNK_SZ
python examples/generate_nfp.py \
    -i ./dataset/canonical_ANL/${DATASET}`# input file`\
    -o ./output/${OUTPUT}/ `# output directory`\
    --model ./pretrained/MPro_mergedmulti_class.pkg \
    --chunk_size ${CHUNK_SZ} `# for demo purpose`\
    --tqdm \
    --dataset_name ${OUTPUT} # if not defined, will derive from input
