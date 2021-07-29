#!/bin/bash

STORAGE=/home/nishida/storage/projects/discourse/CoreferenceResolution/data
ONTONOTES=/home/nishida/storage/dataset/OntoNotes-Release-5.0/ontonotes-release-5.0


##################
# Function
##################


dlx() {
    wget -P ${STORAGE}/$3 $1/$2
import re
    tar -zxvf ${STORAGE}/$3/$2 -C ${STORAGE}/$3
    rm ${STORAGE}/$3/$2
}


##################
# OntoNotes
##################


conll_url=https://conll.cemantix.org/2012/download
dlx ${conll_url} conll-2012-train.v4.tar.gz ontonotes
dlx ${conll_url} conll-2012-development.v4.tar.gz ontonotes
dlx ${conll_url}/test conll-2012-test-key.tar.gz ontonotes
dlx ${conll_url}/test conll-2012-test-official.v9.tar.gz ontonotes
dlx ${conll_url} conll-2012-scripts.v3.tar.gz ontonotes
dlx https://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz ontonotes

bash ./conll-2012/v3/scripts/skeleton2conll.sh -D ${ONTONOTES}/data/files/data ${STORAGE}/ontonotes/conll-2012

mkdir ${STORAGE}/ontonotes-preprocessed
cat ${STORAGE}/ontonotes/conll-2012/v4/data/train/data/english/annotations/*/*/*/*.v4_gold_conll >> ${STORAGE}/ontonotes-preprocessed/ontonotes.train.english.v4_gold_conll
cat ${STORAGE}/ontonotes/conll-2012/v4/data/development/data/english/annotations/*/*/*/*.v4_gold_conll >> ${STORAGE}/ontonotes-preprocessed/ontonotes.dev.english.v4_gold_conll
cat ${STORAGE}/ontonotes/conll-2012/v4/data/test/data/english/annotations/*/*/*/*.v4_gold_conll >> ${STORAGE}/ontonotes-preprocessed/ontonotes.test.english.v4_gold_conll

for seg_len in 384 512
do
    python preprocess1.py \
        --input_dir ${STORAGE}/ontonotes-preprocessed \
        --output_dir ${STORAGE}/ontonotes-preprocessed \
        --dataset_name ontonotes \
        --language english \
        --extension v4_gold_conll \
        --tokenizer_name bert-base-cased \
        --seg_len ${seg_len}
done

for seg_len in 384 512
do
    python preprocess2.py \
        --input_file ${STORAGE}/ontonotes-preprocessed/ontonotes.train.english.${seg_len}.bert-base-cased.jsonlines \
        --is_training 1 \
        --tokenizer_name bert-base-cased \
        --seg_len ${seg_len}

    for split in dev test
    do
        python preprocess2.py \
            --input_file ${STORAGE}/ontonotes-preprocessed/ontonotes.${split}.english.${seg_len}.bert-base-cased.jsonlines \
            --is_training 0 \
            --tokenizer_name bert-base-cased \
            --seg_len ${seg_len}
    done
done


# ##################
# # CRAFT
# ##################


python ./preprocessing/prepare_craft.py
python ./preprocessing/remove_discontinuous_mentions.py

mkdir ${STORAGE}/craft-preprocessed
cat ${STORAGE}/craft-conll/train/*.continuous_only_conll >> ${STORAGE}/craft-preprocessed/craft.train.english.gold_conll
cat ${STORAGE}/craft-conll/dev/*.continuous_only_conll >> ${STORAGE}/craft-preprocessed/craft.dev.english.gold_conll
cat ${STORAGE}/craft-conll/test/*.continuous_only_conll >> ${STORAGE}/craft-preprocessed/craft.test.english.gold_conll

TOKENIZER=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

for seg_len in 384 512
do
    python preprocess1.py \
        --input_dir ${STORAGE}/craft-preprocessed \
        --output_dir ${STORAGE}/craft-preprocessed \
        --dataset_name craft \
        --language english \
        --extension gold_conll \
        --tokenizer ${TOKENIZER} \
        --seg_len ${seg_len}
done

for seg_len in 384 512
do
    python preprocess2.py \
        --input_file ${STORAGE}/craft-preprocessed/craft.train.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
        --is_training 1 \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}

    for split in dev test
    do
        python preprocess2.py \
            --input_file ${STORAGE}/craft-preprocessed/craft.${split}.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
            --is_training 0 \
            --tokenizer_name ${TOKENIZER} \
            --seg_len ${seg_len}
    done
done


##################
# SpanBERT
##################


download_spanbert() {
    model=$1
    wget -P ${STORAGE} https://dl.fbaipublicfiles.com/fairseq/models/${model}.tar.gz
    mkdir ${STORAGE}/${model}
    tar zxvf ${STORAGE}/${model}.tar.gz -C ${STORAGE}/${model}
    rm ${STORAGE}/${model}.tar.gz
}

download_spanbert spanbert_hf_base
download_spanbert spanbert_hf


