#!/usr/bin/env sh

# Docker image
IMAGE=ucdenverccp/craft-eval:4.0.1_0.1.2

CRAFT=/home/norikinishida/storage/dataset/CRAFT.v4
# EVALUATOR=xxx
# SCORER=xxx

GOLD=/home/norikinishida/storage/dataset/CRAFT.v4/coref-conll
PRED=/home/norikinishida/storage/projects/discourse/CoreferenceResolution/results/main.joshi2020_pubmedbert_base_craft

python prepare_for_evaluation.py \
    --gold ${GOLD} \
    --pred ${PRED}

docker run --rm \
    -v ${CRAFT}:/corpus-distribution \
    -v ${PRED}/files-to-evaluate:/files-to-evaluate \
    ${IMAGE} sh -c 'cd /home/craft/evaluation && boot eval-coreference'

python show_results.py \
    --path ${PRED}/files-to-evaluate/coref_results.tsv

