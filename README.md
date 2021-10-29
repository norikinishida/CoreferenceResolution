# coreference-resolution

(c) 2021 Noriki Nishida

This is an implementation of coreference resolution models:

- End-to-end coreference resolution model using BERT/SpanBERT ([Joshi et al., 2019](https://aclanthology.org/D19-1588); [Joshi et al., 2020](https://aclanthology.org/2020.tacl-1.5)])

## Requirements

- numpy
- pytorch
- huggingface
- jsonlines
- pyprind
- https://github.com/norikinishida/utils.git

## Configuration

The following files need to be editted according to your environment.

- config/path.conf
- run_preprocessing.sh

## Preprocessing

```
./run_preprocessing.sh
```

The following files will be generated:

- /path/to/data/ontonotes/
- /path/to/data/ontonotes-preprocessed/
- /path/to/data/craft/
- /path/to/data/craft-conll/
- /path/to/data/craft-preprocessed/
- /path/to/caches/ontonotes.{train,dev,test}.english.{384,512}.bert-base-cased.npy
- /path/to/caches/craft.{train,dev,test}.english.{384,512}.bert-base-cased.npy

## Training

```
python main.py --gpu 0 --config joshi2020_spanbertlarge_ontonotes --actiontype train
```

The following files will be generated:

- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.training.log
- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.training.jsonl
- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.validation.jsonl
- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.model

## Evaluation

```
python main.py --gpu 0 --config joshi2020_spanbertlarge_ontonotes --prefix <date> --actiontype evaluate
```

The following files will be generated:

- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.evaluation.log
- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.evaluation.conll
- /path/to/results/main.joshi2020_spanbertlarge_ontonotes/\<date\>.evaluation.jsonl

### Evaluation on CRAFT using the official docker evaluation script

We ran the following shell script in the ./craft-shared-tasks directory to perform the official craft-shared-task evaluation protocol on a prediction file: e.g, "Jul28_22-24-14.evaluation.conll".

Before running the shell script, you need to edit the paths (e.g., "CRAFT", "PRED", etc.) in the script appropriately.

```
./run_docker_eval.sh
```

The following files will be generated:

- /path/to/results/main.joshi2020_pubmedbertlarge_craft/files-to-evaluate/\<filename\>.conll
- /path/to/results/main.joshi2020_pubmedbertlarge_craft/files-to-evaluate/coref_results.tsv


