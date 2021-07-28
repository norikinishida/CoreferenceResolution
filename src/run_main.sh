#!/usr/bin/env sh

GPU=0
CONFIG=joshi2020_spanbert_large_ontonotes

# Training
python main.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train

# Evaluation
# MYPREFIX=Jul27_13-59-18
# python main.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate

