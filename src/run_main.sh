#!/usr/bin/env sh

GPU=0
CONFIG=joshi2020_spanbertlarge_ontonotes
# CONFIG=joshi2020_spanbertbase_craft
# CONFIG=joshi2020_pubmedbertbase_craft

# Training
python main.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train

# Evaluation
# MYPREFIX=Jul28_17-28-11
# python main.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate
