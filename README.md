# coreference-resolution

This repository is an implementation of coreference resolution models:

- End-to-end coreference resolution model using BERT/SpanBERT ([Joshi et al., 2019](https://aclanthology.org/D19-1588); [Joshi et al., 2020](https://aclanthology.org/2020.tacl-1.5))

## Requirements

- numpy
- pytorch
- huggingface
- jsonlines
- pyprind
- https://github.com/norikinishida/utils

## Configuration

The following files need to be editted according to your environment.

- `config/path.conf`
- `run_preprocessing.sh`

## Preprocessing

Please see `./run_preprocessing.sh` for details.

Outputs:

- Preprocessed datasets: `<caches>/{ontonotes,craft}.{train,dev,test}.english.{384,512}.bert-base-cased.npy`
- Gold annotations: `<caches>/{ontonotes,craft}.{train,dev,test}.english.{v4_gold_conll,gold_conll,gold_original_conll}`

`<caches>` is specified in `./config/path.conf`.

## Training

Experiment configurations are found in `./config` (e.g., `joshi2020.conf`).
You can also add your own configuration.
Choose a configuration name (e.g., `joshi2020_spanbertlarge_ontonotes`), and run

```
python main.py --gpu <gpu_id> --config <config_name> --actiontype train
```

The following command is an example to train an end-to-end CR model (Joshi+, 2020) using SpanBERT (large) on OntoNotes:

```
python main.py --gpu 0 --config joshi2020_spanbertlarge_ontonotes --actiontype train
```

The results are stored in the `<results>/main/<config_name>` directory.
`<results>` is specified in `./config/path.conf`.

Outputs:
- Log: `<results>/main/<config_name>/<prefix>.training.log`
- Training losses: `<results>/main/<config_name>/<prefix>.train.losses.jsonl`
- Model parameters: `<results>/main/<config_name>/<prefix>.model`
- Validation scores: `<results>/main/<config_name>/<prefix>.dev.eval.jsonl`

`<prefix>` is automatically determined based on the execution time, .e.g, `Jun09_01-23-45`.

## Evaluation

The trained model can be evaluated on the test dataset using the following command:

```
python main.py --gpu <gpu_id> --config <config_name> --prefix <prefix> --actiontype evaluate
```

The following command is an example to evaluate the above model on the OntoNotes test set:

```
python main.py --gpu 0 --config joshi2020_spanbertlarge_ontonotes --prefix Jun09_01-23-45 --actiontype evaluate
```

Results are stored in the `<results>/main/<config_name>` directory.

Outputs:

- Log: `<results>/main/<config_name>/<prefix>.evaluation.log`
- Evaluation outputs (CoNLL format): `<results>/main/<config_name>/<prefix>.test.pred.conll`
- Evaluation outputs (JSON format): `<results>/main/<config_name>/<prefix>.test.pred.clusters`
- Evaluation scores: `<results>/main/<config_name>/<prefix>.test.eval.json`

### Evaluation on CRAFT using the official docker evaluation script

We ran the following shell script in the `./craft-shared-tasks` directory to perform the official craft-shared-task evaluation protocol on a prediction file: e.g, `Jul09_01-23-45.evaluation.conll`.

Before running the shell script, you need to edit the paths (e.g., `CRAFT`, `PRED`, etc.) in the script appropriately, and run

```
./run_docker_eval.sh
```

Outputs:

- `<results>/main/<config_name>/files-to-evaluate/*.conll`
- `<results>/main/<config_name>/files-to-evaluate/coref_results.tsv`


