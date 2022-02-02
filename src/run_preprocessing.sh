#!/bin/bash

ONTONOTES=/home/nishida/storage/dataset/OntoNotes-Release-5.0/ontonotes-release-5.0
ACL=/home/nishida/storage/dataset/ACL-Coref/Coreference_annotated_corpus/head_auto_conlls

STORAGE=/home/nishida/storage/projects/discourse/coreference-resolution
STORAGE_DATA=${STORAGE}/data
STORAGE_CACHES=${STORAGE}/caches


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


##################
# Function
##################


dlx() {
    wget -P ${STORAGE_DATA}/$3 $1/$2
    tar -zxvf ${STORAGE_DATA}/$3/$2 -C ${STORAGE_DATA}/$3
    rm ${STORAGE_DATA}/$3/$2
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

bash ./conll-2012/v3/scripts/skeleton2conll.sh -D ${ONTONOTES}/data/files/data ${STORAGE_DATA}/ontonotes/conll-2012

mkdir ${STORAGE_DATA}/ontonotes-preprocessed
cat ${STORAGE_DATA}/ontonotes/conll-2012/v4/data/train/data/english/annotations/*/*/*/*.v4_gold_conll >> ${STORAGE_DATA}/ontonotes-preprocessed/ontonotes.train.english.v4_gold_conll
cat ${STORAGE_DATA}/ontonotes/conll-2012/v4/data/development/data/english/annotations/*/*/*/*.v4_gold_conll >> ${STORAGE_DATA}/ontonotes-preprocessed/ontonotes.dev.english.v4_gold_conll
cat ${STORAGE_DATA}/ontonotes/conll-2012/v4/data/test/data/english/annotations/*/*/*/*.v4_gold_conll >> ${STORAGE_DATA}/ontonotes-preprocessed/ontonotes.test.english.v4_gold_conll

TOKENIZER=bert-base-cased

for seg_len in 384 512
do
    python ./preprocessing/preprocess1.py \
        --input_dir ${STORAGE_DATA}/ontonotes-preprocessed \
        --output_dir ${STORAGE_DATA}/ontonotes-preprocessed \
        --dataset_name ontonotes \
        --language english \
        --extension v4_gold_conll \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}
done

for seg_len in 384 512
do
    python ./preprocessing/preprocess2.py \
        --input_file ${STORAGE_DATA}/ontonotes-preprocessed/ontonotes.train.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
        --is_training 1 \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}

    for split in dev test
    do
        python ./preprocessing/preprocess2.py \
            --input_file ${STORAGE_DATA}/ontonotes-preprocessed/ontonotes.${split}.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
            --is_training 0 \
            --tokenizer_name ${TOKENIZER} \
            --seg_len ${seg_len}
    done
done

cp ${STORAGE_DATA}/ontonotes-preprocessed/*.v4_gold_conll ${STORAGE_CACHES}/


##################
# CRAFT
##################


python ./preprocessing/prepare_craft.py
python ./preprocessing/remove_discontinuous_mentions.py

mkdir ${STORAGE_DATA}/craft-preprocessed
cat ${STORAGE_DATA}/craft-conll/train/*.continuous_only_conll >> ${STORAGE_DATA}/craft-preprocessed/craft.train.english.gold_conll
cat ${STORAGE_DATA}/craft-conll/dev/*.continuous_only_conll >> ${STORAGE_DATA}/craft-preprocessed/craft.dev.english.gold_conll
cat ${STORAGE_DATA}/craft-conll/test/*.continuous_only_conll >> ${STORAGE_DATA}/craft-preprocessed/craft.test.english.gold_conll

TOKENIZER=bert-base-cased
# TOKENIZER=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

for seg_len in 384 512
do
    python ./preprocessing/preprocess1.py \
        --input_dir ${STORAGE_DATA}/craft-preprocessed \
        --output_dir ${STORAGE_DATA}/craft-preprocessed \
        --dataset_name craft \
        --language english \
        --extension gold_conll \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}
done

for seg_len in 384 512
do
    python ./preprocessing/preprocess2.py \
        --input_file ${STORAGE_DATA}/craft-preprocessed/craft.train.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
        --is_training 1 \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}

    for split in dev test
    do
        python ./preprocessing/preprocess2.py \
            --input_file ${STORAGE_DATA}/craft-preprocessed/craft.${split}.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
            --is_training 0 \
            --tokenizer_name ${TOKENIZER} \
            --seg_len ${seg_len}
    done
done

cp ${STORAGE_DATA}/craft-preprocessed/*.gold_conll ${STORAGE_CACHES}/
cat ${STORAGE_DATA}/craft-conll/train/*.conll >> ${STORAGE_CACHES}/craft.train.english.gold_original_conll
cat ${STORAGE_DATA}/craft-conll/dev/*.conll >> ${STORAGE_CACHES}/craft.dev.english.gold_original_conll
cat ${STORAGE_DATA}/craft-conll/test/*.conll >> ${STORAGE_CACHES}/craft.test.english.gold_original_conll


##################
# ACL-Coref
##################


mkdir ${STORAGE_DATA}/acl-preprocessed
cat ${ACL}/train/*.auto_conll >> ${STORAGE_DATA}/acl-preprocessed/acl.train.english.gold_conll
cat ${ACL}/dev/*.auto_conll >> ${STORAGE_DATA}/acl-preprocessed/acl.dev.english.gold_conll
cat ${ACL}/test/*.auto_conll >> ${STORAGE_DATA}/acl-preprocessed/acl.test.english.gold_conll

TOKENIZER=bert-base-cased

for seg_len in 384 512
do
    python ./preprocessing/preprocess1.py \
        --input_dir ${STORAGE_DATA}/acl-preprocessed \
        --output_dir ${STORAGE_DATA}/acl-preprocessed \
        --dataset_name acl \
        --language english \
        --extension gold_conll \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}
done

for seg_len in 384 512
do
    python ./preprocessing/preprocess2.py \
        --input_file ${STORAGE_DATA}/acl-preprocessed/acl.train.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
        --is_training 1 \
        --tokenizer_name ${TOKENIZER} \
        --seg_len ${seg_len}

    for split in dev test
    do
        python ./preprocessing/preprocess2.py \
            --input_file ${STORAGE_DATA}/acl-preprocessed/acl.${split}.english.${seg_len}.`basename ${TOKENIZER}`.jsonlines \
            --is_training 0 \
            --tokenizer_name ${TOKENIZER} \
            --seg_len ${seg_len}
    done
done

cp ${STORAGE_DATA}/acl-preprocessed/*.gold_conll ${STORAGE_CACHES}/


