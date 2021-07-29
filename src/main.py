import argparse
import os
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW
import jsonlines
import pyprind

import utils

import systems
from metrics import CorefEvaluator
import conll


def main(args):
    ##################
    # Arguments
    ##################

    device = torch.device(f"cuda:{args.gpu}")
    # path_hyperparams = args.hyperparams
    config_name = args.config
    prefix = args.prefix
    actiontype = args.actiontype

    # Check
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()

    assert actiontype in ["train", "evaluate"]

    ##################
    # Config
    ##################

    # config = utils.Config(path_hyperparams)
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name=config_name)
    sw = utils.StopWatch()
    sw.start("main")

    ##################
    # Path setting
    ##################

    base_dir = "main.%s" % \
              (# utils.get_basename_without_ext(path_hyperparams),
               config_name)

    utils.mkdir(os.path.join(config["results"], base_dir))

    path_log = None

    # Used in training
    if actiontype == "train":
        path_log = os.path.join(config["results"], base_dir, prefix + ".training.log")
    path_train_jsonl = os.path.join(config["results"], base_dir, prefix + ".training.jsonl")
    path_valid_jsonl = os.path.join(config["results"], base_dir, prefix + ".validation.jsonl")
    path_snapshot = os.path.join(config["results"], base_dir, prefix + ".model")

    # Used in evaluation
    if actiontype == "evaluate":
        path_log = os.path.join(config["results"], base_dir, prefix + ".evaluation.log")
    path_pred = os.path.join(config["results"], base_dir, prefix + ".evaluation.conll")
    if config["dataset"] == "ontonotes":
        path_gold = os.path.join(config["data"], "ontonotes-preprocessed", "ontonotes.test.english.v4_gold_conll")
    elif config["dataset"] == "craft":
        path_gold = os.path.join(config["data"], "craft-preprocessed", "craft.test.english.gold_conll")
    else:
        raise Exception("Never occur.")
    path_eval = os.path.join(config["results"], base_dir, prefix + ".evaluation.json")

    utils.set_logger(path_log)

    utils.writelog("device: %s" % device)
    # utils.writelog("model_name: %s" % model_name)
    # utils.writelog("path_hyperparams: %s" % path_hyperparams)
    utils.writelog("config_name: %s" % config_name)
    utils.writelog("prefix: %s" % prefix)
    utils.writelog("actiontype: %s" % actiontype)

    utils.writelog(config)

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_jsonl: %s" % path_train_jsonl)
    utils.writelog("path_valid_jsonl: %s" % path_valid_jsonl)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_pred: %s" % path_pred)
    utils.writelog("path_gold: %s" % path_gold)
    utils.writelog("path_eval: %s" % path_eval)

    ##################
    # Random seed
    ##################

    random_seed = 1234
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    utils.writelog("random_seed: %d" % random_seed)

    ##################
    # Data preparation
    ##################

    sw.start("data")

    if config["dataset"] == "ontonotes":
        train_dataset = np.load(os.path.join(config["caches"], f"ontonotes.train.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.npy"), allow_pickle=True)
        dev_dataset = np.load(os.path.join(config["caches"], f"ontonotes.dev.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches"], f"ontonotes.test.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.npy"), allow_pickle=True)
    elif config["dataset"] == "craft":
        train_dataset = np.load(os.path.join(config["caches"], f"craft.train.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.npy"), allow_pickle=True)
        dev_dataset = np.load(os.path.join(config["caches"], f"craft.dev.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches"], f"craft.test.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.npy"), allow_pickle=True)
    else:
        raise Exception("Never occur.")

    utils.writelog("Number of training data: %d" % len(train_dataset))
    utils.writelog("Number of validation data: %d" % len(dev_dataset))
    utils.writelog("Number of test data: %d" % len(test_dataset))

    sw.stop("data")
    utils.writelog("Loaded the corpus. %f sec." % sw.get_time("data"))

    ##################
    # System preparation
    ##################

    system = systems.CorefSystem(
                            device=device,
                            config=config)

    # Load pre-trained parameters
    if actiontype != "train":
        system.load_model(path=path_snapshot)
        utils.writelog("Loaded model from %s" % path_snapshot)

    system.to_gpu(device=device)

    ##################
    # Training / evaluation
    ##################

    if actiontype == "train":
        train(config=config,
              system=system,
              train_dataset=train_dataset,
              dev_dataset=dev_dataset,
              path_train_jsonl=path_train_jsonl,
              path_valid_jsonl=path_valid_jsonl,
              path_snapshot=path_snapshot,
              path_pred=None,
              path_gold=None)

    elif actiontype == "evaluate":
        with torch.no_grad():
            scores = evaluate(config=config,
                              system=system,
                              dataset=test_dataset,
                              step=0,
                              official=True,
                              path_pred=path_pred,
                              path_gold=path_gold)
            utils.write_json(path_eval, scores)
            utils.writelog(utils.pretty_format_dict(scores))

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_jsonl: %s" % path_train_jsonl)
    utils.writelog("path_valid_jsonl: %s" % path_valid_jsonl)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_pred: %s" % path_pred)
    utils.writelog("path_gold: %s" % path_gold)
    utils.writelog("path_eval: %s" % path_eval)
    utils.writelog("Done.")
    sw.stop("main")
    utils.writelog("Time: %f min." % sw.get_time("main", minute=True))


#####################################
# Training
#####################################



def train(config,
          system,
          train_dataset,
          dev_dataset,
          path_train_jsonl,
          path_valid_jsonl,
          path_snapshot,
          path_pred=None,
          path_gold=None):
    """
    Parameters
    ----------
    config: utils.Config
    system: System
    train_dataset: numpy.ndarray
    dev_dataset: numpy.ndarray
    path_train_jsonl: str
    path_valid_jsonl: str
    path_snapshot: str
    path_pred: str, default None
    path_gold: str, default None
    """
    torch.autograd.set_detect_anomaly(True)

    # Get optimizers and schedulers
    n_train = len(train_dataset)
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    total_update_steps = n_train * max_epoch // batch_size
    warmup_steps = int(total_update_steps * config["warmup_ratio"])

    optimizers = get_optimizer(model=system.model, config=config)
    schedulers = get_scheduler(optimizers=optimizers, total_update_steps=total_update_steps, warmup_steps=warmup_steps)

    utils.writelog("********************Training********************")
    utils.writelog("n_train: %d" % n_train)
    utils.writelog("max_epoch: %d" % max_epoch)
    utils.writelog("batch_size: %d" % batch_size)
    utils.writelog("total_update_steps: %d" % total_update_steps)
    utils.writelog("warmup_steps: %d" % warmup_steps)

    writer_train = jsonlines.Writer(open(path_train_jsonl, "w"), flush=True)
    writer_valid = jsonlines.Writer(open(path_valid_jsonl, "w"), flush=True)
    bestscore_holder = utils.BestScoreHolder(scale=1.0)
    bestscore_holder.init()
    step = 0
    bert_param, task_param = system.model.get_params()

    #################
    # Initial validation phase
    #################

    with torch.no_grad():
        scores = evaluate(config=config,
                          system=system,
                          dataset=dev_dataset,
                          step=step,
                          official=False,
                          path_pred=path_pred,
                          path_gold=path_gold)
        scores["step"] = 0
        writer_valid.write(scores)
        utils.writelog(utils.pretty_format_dict(scores))

        bestscore_holder.compare_scores(scores["Average F1 (py)"], 0)

        system.save_model(path=path_snapshot)
        utils.writelog("Saved model to %s" % path_snapshot)

    #################
    # /Initial validation phase
    #################

    system.model.zero_grad()
    for epoch in range(1, max_epoch+1):

        #################
        # Training phase
        #################

        random.shuffle(train_dataset)

        batch_loss = []
        for data in train_dataset:

            #################
            # One data
            #################

            # Forward
            loss = system.compute_loss(data=data)
            if batch_size > 1.0:
                loss = loss / batch_size

            # Backward
            loss.backward()
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(bert_param, config["max_grad_norm"])
                torch.nn.utils.clip_grad_norm_(task_param, config["max_grad_norm"])

            batch_loss.append(loss.item())

            # Update?
            if len(batch_loss) % batch_size == 0:
                # Update
                for optimizer in optimizers:
                    optimizer.step()
                system.model.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()

                step += 1

                # Report
                avg_loss = np.mean(batch_loss).item()
                out = {"step": step,
                       "epoch": epoch,
                       "progress": "%d/%d" % (step * batch_size, n_train),
                       "progress_ratio": float(step * batch_size) / n_train * 100.0,
                       "avg_loss": avg_loss,
                       "max_valid_score": bestscore_holder.best_score,
                       "patience": bestscore_holder.patience}
                writer_train.write(out)
                utils.writelog(utils.pretty_format_dict(out))

                batch_loss = []

                #################
                # Validation phase
                #################

                if step > 0 and step % config["valid_frequency"] == 0:
                    with torch.no_grad():
                        scores = evaluate(config=config,
                                          system=system,
                                          dataset=dev_dataset,
                                          step=step,
                                          official=False,
                                          path_pred=path_pred,
                                          path_gold=path_gold)
                        scores["step"] = step
                        writer_valid.write(scores)
                        utils.writelog(utils.pretty_format_dict(scores))

                    did_update = bestscore_holder.compare_scores(scores["Average F1 (py)"], step)
                    utils.writelog("[Step %d] Validation max F1: %f" % (step, bestscore_holder.best_score))

                    if did_update:
                        system.save_model(path=path_snapshot)
                        utils.writelog("Saved model to %s" % path_snapshot)

                #################
                # /Validation phase
                #################

            #################
            # /One data
            #################

        #################
        # /Training phase
        #################

    writer_train.close()
    writer_valid.close()


def get_optimizer(model, config):
    """
    Parameters
    ----------
    model: CorefModel
    config: utils.Config

    Returns
    -------
    [transformers.AdamW, torch.optim.Adam]
    """
    no_decay = ['bias', 'LayerNorm.weight']
    bert_param, task_param = model.get_params(named=True)
    grouped_bert_param = [
        {
            'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
            'lr': config['bert_learning_rate'],
            'weight_decay': config['adam_weight_decay'],
        },
        {
            'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
            'lr': config['bert_learning_rate'],
            'weight_decay': 0.0
        }
    ]
    optimizers = [
        AdamW(grouped_bert_param, lr=config['bert_learning_rate'], eps=config['adam_eps']), # NOTE: Comment out this line for freezing SpanBERT
        Adam(model.get_params()[1], lr=config['task_learning_rate'], eps=config['adam_eps'], weight_decay=0)
    ]
    return optimizers


def get_scheduler(optimizers, total_update_steps, warmup_steps):
    """
    Parameters
    ----------
    optimizers: [transformers.AdamW, torch.optim.Adam]
    total_update_steps: int
    warmup_steps: int

    Returns
    -------
    list[torch.optim.lr_scheduler.LambdaLR]
    """
    # Only warm up bert lr
    def lr_lambda_bert(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

    def lr_lambda_task(current_step):
        return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

    # NOTE: Modified for freezing SpanBERT
    schedulers = [
        LambdaLR(optimizers[0], lr_lambda_bert),
        LambdaLR(optimizers[1], lr_lambda_task)
    ]
    # schedulers = [
    #     LambdaLR(optimizers[0], lr_lambda_task)
    # ]
    return schedulers
    # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])


#####################################
# Evaluation
#####################################


def evaluate(config,
             system,
             dataset,
             step,
             official,
             path_pred,
             path_gold):
    """
    Parameters
    ----------
    config: utils.Config
    system: System
    dataset: numpy.ndarray
    step: int
    official: bool
    path_pred: str
    path_gold: str

    Returns
    -------
    dict[str, Any]
    """
    utils.writelog("[Step %d] evaluating on %d samples..." % (step, len(dataset)))
    sw = utils.StopWatch()
    sw.start()

    scores = {}

    evaluator = CorefEvaluator()
    doc_to_prediction = {}
    for data in pyprind.prog_bar(dataset):
        # data = data[:7] # Strip out gold (cf., old code)
        (predicted_mentions, predicted_clusters, antecedents), evaluator = \
                system.predict(data=data, evaluator=evaluator, gold_clusters=data.gold_clusters)
        doc_to_prediction[data.doc_key] = predicted_clusters

    # 公式スクリプトによる評価
    if official:
        subtoken_maps = {data.doc_key: data.subtoken_map for data in dataset}
        conll_results = conll.evaluate_conll(gold_path=path_gold,
                                             pred_path=path_pred,
                                             predictions=doc_to_prediction,
                                             subtoken_maps=subtoken_maps,
                                             official_stdout=True)
        official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        scores["Average F1 (conll)"] = official_f1

    precision, recall, f1 = evaluator.get_prf()
    scores["Average precision (py)"] = precision * 100.0
    scores["Average recall (py)"] = recall * 100.0
    scores["Average F1 (py)"] = f1 * 100.0

    return scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--hyperparams", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)
