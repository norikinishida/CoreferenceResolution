#!/usr/bin/env sh

GPU=0
CONFIG=joshi2020_spanbertlarge_ontonotes
# CONFIG=joshi2020_spanbertbase_craft
# CONFIG=joshi2020_pubmedbertbase_craft

python main.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train_and_evaluate

