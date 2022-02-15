import argparse
import os
import random

import numpy as np
import torch
import jsonlines
import pyprind

import utils

import shared_functions
import systems
import metrics
import conll


def main(args):
    ##################
    # Arguments
    ##################

    device = torch.device(f"cuda:{args.gpu}")
    config_name = args.config
    prefix = args.prefix
    actiontype = args.actiontype

    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()

    assert actiontype in ["train", "evaluate"]

    ##################
    # Config
    ##################

    config = utils.get_hocon_config(config_path="./config/main.conf", config_name=config_name)
    sw = utils.StopWatch()
    sw.start("main")

    ##################
    # Path setting
    ##################

    base_dir = "main.%s" % config_name

    utils.mkdir(os.path.join(config["results"], base_dir))

    # Log file
    path_log = None
    if actiontype == "train":
        path_log = os.path.join(config["results"], base_dir, prefix + ".training.log")
    elif actiontype == "evaluate":
        path_log = os.path.join(config["results"], base_dir, prefix + ".evaluation.log")

    # Training loss and etc.
    path_train_jsonl = os.path.join(config["results"], base_dir, prefix + ".training.jsonl")

    # Validation outputs and scores
    path_valid_pred = os.path.join(config["results"], base_dir, prefix + ".validation.conll")
    path_valid_jsonl = os.path.join(config["results"], base_dir, prefix + ".validation.jsonl")

    # Model snapshot
    path_snapshot = os.path.join(config["results"], base_dir, prefix + ".model")

    # Evaluation outputs and scores
    path_test_pred = os.path.join(config["results"], base_dir, prefix + ".evaluation.conll")
    path_test_json = os.path.join(config["results"], base_dir, prefix + ".evaluation.json")

    # Gold data for validation and evaluation
    if config["dataset"] == "ontonotes":
        path_valid_gold = os.path.join(config["caches"], "ontonotes.dev.english.v4_gold_conll")
        path_test_gold = os.path.join(config["caches"], "ontonotes.test.english.v4_gold_conll")
    elif config["dataset"] == "craft":
        path_valid_gold = os.path.join(config["caches"], "craft.dev.english.gold_conll")
        path_test_gold = os.path.join(config["caches"], "craft.test.english.gold_conll")
    else:
        raise Exception("Never occur.")

    utils.set_logger(path_log)

    utils.writelog("device: %s" % device)
    utils.writelog("config_name: %s" % config_name)
    utils.writelog("prefix: %s" % prefix)
    utils.writelog("actiontype: %s" % actiontype)

    utils.writelog(config)

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_jsonl: %s" % path_train_jsonl)
    utils.writelog("path_valid_pred: %s" % path_valid_pred)
    utils.writelog("path_valid_gold: %s" % path_valid_gold)
    utils.writelog("path_valid_jsonl: %s" % path_valid_jsonl)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_test_pred: %s" % path_test_pred)
    utils.writelog("path_test_gold: %s" % path_test_gold)
    utils.writelog("path_test_json: %s" % path_test_json)

    ##################
    # Data preparation
    ##################

    sw.start("data")

    if config["dataset"] == "ontonotes":
        if config["model_name"] == "joshi2020_discdep01":
            train_dataset = np.load(os.path.join(config["caches"], f"ontonotes.train.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.with_spl.npy"), allow_pickle=True)
            dev_dataset = np.load(os.path.join(config["caches"], f"ontonotes.dev.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.with_spl.npy"), allow_pickle=True)
            test_dataset = np.load(os.path.join(config["caches"], f"ontonotes.test.english.{config['max_segment_len']}.{os.path.basename(config['bert_tokenizer_name'])}.with_spl.npy"), allow_pickle=True)
        else:
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
    # Action
    ##################

    if actiontype == "train":
        train(config=config,
              system=system,
              train_dataset=train_dataset,
              dev_dataset=dev_dataset,
              path_train_jsonl=path_train_jsonl,
              path_valid_pred=None,
              path_valid_gold=None,
              path_valid_jsonl=path_valid_jsonl,
              path_snapshot=path_snapshot)

    elif actiontype == "evaluate":
        with torch.no_grad():
            scores = evaluate(config=config,
                              system=system,
                              dataset=test_dataset,
                              step=0,
                              official=True,
                              path_pred=path_test_pred,
                              path_gold=path_test_gold)
            utils.write_json(path_test_json, scores)
            utils.writelog(utils.pretty_format_dict(scores))

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_jsonl: %s" % path_train_jsonl)
    utils.writelog("path_valid_pred: %s" % path_valid_pred)
    utils.writelog("path_valid_gold: %s" % path_valid_gold)
    utils.writelog("path_valid_jsonl: %s" % path_valid_jsonl)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_test_pred: %s" % path_test_pred)
    utils.writelog("path_test_gold: %s" % path_test_gold)
    utils.writelog("path_test_json: %s" % path_test_json)
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
          path_valid_pred,
          path_valid_gold,
          path_valid_jsonl,
          path_snapshot):
    """
    Parameters
    ----------
    config: utils.Config
    system: System
    train_dataset: numpy.ndarray
    dev_dataset: numpy.ndarray
    path_train_jsonl: str
    path_valid_pred: str or None
    path_valid_gold: str or None
    path_valid_jsonl: str
    path_snapshot: str
    """
    torch.autograd.set_detect_anomaly(True)

    # Get optimizers and schedulers
    n_train = len(train_dataset)
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    total_update_steps = n_train * max_epoch // batch_size
    warmup_steps = int(total_update_steps * config["warmup_ratio"])

    optimizers = shared_functions.get_optimizer(model=system.model, config=config)
    schedulers = shared_functions.get_scheduler(optimizers=optimizers, total_update_steps=total_update_steps, warmup_steps=warmup_steps)

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
                          path_pred=path_valid_pred,
                          path_gold=path_valid_gold)
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
                                          path_pred=path_valid_pred,
                                          path_gold=path_valid_gold)
                        scores["step"] = step
                        writer_valid.write(scores)
                        utils.writelog(utils.pretty_format_dict(scores))

                    did_update = bestscore_holder.compare_scores(scores["Average F1 (py)"], step)
                    utils.writelog("[Step %d] Max validation F1: %f" % (step, bestscore_holder.best_score))

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

    evaluator = metrics.CorefEvaluator()
    doc_to_prediction = {}
    doc_to_prediction_text = {}
    for data in pyprind.prog_bar(dataset):
        predicted_clusters, evaluator = \
                system.predict(data=data, evaluator=evaluator, gold_clusters=data.gold_clusters)
        if not official:
            # for time saving
            doc_to_prediction[data.doc_key] = predicted_clusters # unused
            doc_to_prediction_text[data.doc_key] = predicted_clusters # unused
        else:
            # NOTE: Multiple subtoken-level spans can be the same word-level span.

            # Merge clusters if any clusters have common mentions
            merged_clusters = []
            for cluster in predicted_clusters:
                existing = None
                for mention in cluster:
                    start, end = mention
                    start, end = data.subtoken_map[start], data.subtoken_map[end]
                    for merged_cluster in merged_clusters:
                        for mention2 in merged_cluster:
                            start2, end2 = mention2
                            start2, end2 = data.subtoken_map[start2], data.subtoken_map[end2]
                            if start == start2 and end == end2:
                                existing = merged_cluster
                                break
                        if existing is not None:
                            break
                    if existing is not None:
                        break
                if existing is not None:
                    utils.writelog("[doc_key: %s] Merging clusters" % data.doc_key)
                    existing.update(cluster)
                else:
                    merged_clusters.append(set(cluster))
            merged_clusters = [list(cluster) for cluster in merged_clusters]

            # Merge redundant mentions
            new_predicted_clusters = []
            subtokens = utils.flatten_lists(data.segments)
            for cluster in merged_clusters:
                new_cluster = []
                history = []
                for start_sub, end_sub in sorted(cluster, key=lambda tpl: tpl[1] - tpl[0]):
                    start_word, end_word = data.subtoken_map[start_sub], data.subtoken_map[end_sub]
                    existing = None
                    for start_sub2, end_sub2, start_word2, end_word2 in history:
                        if start_word == start_word2 and end_word == end_word2:
                            existing = (start_sub2, end_sub2, start_word2, end_word2)
                            break
                    if existing is None:
                        history.append((start_sub, end_sub, start_word, end_word))
                        new_cluster.append((start_sub, end_sub))
                    else:
                        start_sub2, end_sub2, _, _ = existing
                        utils.writelog("[doc_key: %s] Merging mentions: '%s' and '%s'" % (data.doc_key, subtokens[start_sub: end_sub + 1], subtokens[start_sub2: end_sub2 + 1]))
                new_cluster = sorted(new_cluster, key=lambda tpl: tpl)
                new_predicted_clusters.append(new_cluster)

            doc_to_prediction[data.doc_key] = new_predicted_clusters
            doc_to_prediction_text[data.doc_key] = [[" ".join(subtokens[begin: end + 1]) for begin, end in cluster] for cluster in new_predicted_clusters]

    # Evaluate by the official script in the CoNLL 2012 shared task
    if official:
        subtoken_maps = {data.doc_key: data.subtoken_map for data in dataset}
        conll_results = conll.evaluate_conll(gold_path=path_gold,
                                             pred_path=path_pred,
                                             predictions=doc_to_prediction,
                                             subtoken_maps=subtoken_maps,
                                             official_stdout=True)
        scores["MUC"] = conll_results["muc"]
        scores["B^3"] = conll_results["bcub"]
        scores["CEAFe"]= conll_results["ceafe"]
        scores["Average F1 (conll)"] = sum(results["f"] for results in conll_results.values()) / len(conll_results)

        # We also save clusters in json format
        # new_doc_to_prediction = {} # dict[str, list[list[(int, int)]]]
        # for doc_key, clusters in doc_to_prediction.items():
        #     new_doc_to_prediction[doc_key] = [] # list[list[(int, int)]]
        #     for cluster in clusters:
        #         new_cluster = [] # list[(int, int)]
        #         for start, end in cluster:
        #             start, end = subtoken_maps[doc_key][start], subtoken_maps[doc_key][end] # word-level span
        #             new_cluster.append((start, end))
        #         assert len(new_cluster) == len(set(new_cluster))
        #         new_doc_to_prediction[doc_key].append(new_cluster)
        # utils.write_json(path_pred.replace(".conll", ".clusters"), new_doc_to_prediction)
        utils.write_json(path_pred.replace(".conll", ".clusters"), doc_to_prediction)
        utils.write_json(path_pred.replace(".conll", ".clusters_text"), doc_to_prediction_text)

    precision, recall, f1 = evaluator.get_prf()
    scores["Average precision (py)"] = precision * 100.0
    scores["Average recall (py)"] = recall * 100.0
    scores["Average F1 (py)"] = f1 * 100.0

    sw.stop()
    utils.writelog("Evaluated %d documents; Time: %f sec." % (len(dataset), sw.get_time()))

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)
