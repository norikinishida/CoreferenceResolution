# CoreferenceResolution

(c) 2021 Noriki Nishida

This is an implementation of coreference resolution systems.

## Implemented systems and datasets

- Supervised coreference resolution using BERT-based end-to-end model (Joshi et al., 2020)
- Datasets
    - OntoNotes, CoNLL 2012 shared task (Pradhan et al., 2012)
    - CRAFT (Cohen et al., 2017)

## Requirements

- numpy
- pytorch
- huggingface
- jsonlines
- pyprind
- https://github.com/norikinishida/utils.git

## Configuration

You need to edit the following files according to your environment.

- config/main.conf
- run_preprocessing.sh

## Preprocessing

```
./run_preprocessing.sh
```

The following files will be generated:
    - /path/to/data/craft/
    - /path/to/data/craft-conll/
    - /path/to/data/craft-preprocessed/
    - /path/to/data/ontonotes/
    - /path/to/data/ontonotes-preprocessed/
    - /path/to/caches/ontonotes.{train,dev,test}.english.{384,512}.<pretrained-bert-name>.npy
    - /path/to/caches/craft.{train,dev,test}.english.{384,512}.<pretrained-bert-name>.npy

In our code, we used SpanBERT (Joshi et al., 2020) for OntoNotes and PubMedBERT (Gu et al., 2020) for CRAFT.
Pre-trained BERT models are specified in ./config/main.conf.

## Training

```
python main.py --gpu 0 --config joshi2020_spanbert_large_craft --actiontype train
```

If you train the system on OntoNotes, change the config name to "joshi2020_spanbert_large_ontonotes".
Details can be found in ./run_main.sh.

The following files will be generated:
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.training.log
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.training.jsonl
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.validation.jsonl
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.model

## Evaluation

```
python main.py --gpu 0 --config joshi2020_spanbert_large_craft --prefix <please_specify_prefix> --actiontype evaluate
```

Details can be found in ./run_main.sh.

The following files will be generated:
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.evaluation.log
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.evaluation.conll
    - /path/to/results/main.joshi2020_spanbert_large_craft/<date>.evaluation.jsonl

## Evaluation on CRAFT using the official docker evaluation script

We ran the following shell script in the ```src/craft-shared-tasks``` directory to perform the official craft-shared-task evaluation protocol on a prediction file: e.g, "Jul28_22-24-14.evaluation.conll".

Before running the shell script, you need to edit the paths (e.g., "CRAFT", "PRED", etc.) in the script appropriately.

```
./run_docker_eval.sh
```

The following files will be generated:
    - /path/to/results/main.joshi2020_spanbert_large_craft/files-to-evaluate/<filename>.conll
    - /path/to/results/main.joshi2020_spanbert_large_craft/files-to-evaluate/coref_results.tsv


