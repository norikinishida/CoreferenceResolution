#!/usr/bin/env sh

GPU=0
CONFIG=joshi2020_spanbert_large_ontonotes
# CONFIG=joshi2020_pubmedbert_large_craft

# Training
python main.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train

# Evaluation
# MYPREFIX=Jul28_17-28-11
# MYPREFIX=Jul28_22-24-14
# python main.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate

