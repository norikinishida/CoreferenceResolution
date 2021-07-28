import argparse
import json
import os

import numpy as np

import utils

import util


def main(args):
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="base_hyperparams")

    input_file = args.input_file
    if args.is_training == 0:
        is_training = False
    else:
        is_training = True
    tokenizer = util.get_tokenizer(args.tokenizer_name)
    max_seg_len = args.seg_len

    genre_dict = {genre: idx for idx, genre in enumerate(config["genres"])}

    dataset = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            # 1データの読み込み
            json_data = json.loads(line)

            doc_key = json_data["doc_key"]

            # Mentions and clusters
            clusters = json_data["clusters"]
            gold_mentions = sorted(tuple(mention) for mention in util.flatten(clusters))
            gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)} # span -> index
            gold_mention_cluster_map = np.zeros(len(gold_mentions)) # 0: no cluster
            for cluster_id, cluster in enumerate(clusters):
                for mention in cluster:
                    gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1

            # Speakers
            speakers = json_data["speakers"]
            speaker_dict = get_speaker_dict(util.flatten(speakers), config["max_num_speakers"])

            # Sentences/segments
            sentences = json_data["sentences"] # Segments
            sentence_map = json_data["sentence_map"]
            num_words = sum([len(s) for s in sentences])
            sentence_len = np.array([len(s) for s in sentences])

            # BERT input IDs/mask, speaker IDs
            input_ids, input_mask, speaker_ids = [], [], []
            for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
                sent_input_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
                sent_input_mask = [1] * len(sent_input_ids)
                sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
                while len(sent_input_ids) < max_seg_len:
                    sent_input_ids.append(0)
                    sent_input_mask.append(0)
                    sent_speaker_ids.append(0)
                input_ids.append(sent_input_ids)
                input_mask.append(sent_input_mask)
                speaker_ids.append(sent_speaker_ids)
            input_ids = np.array(input_ids)
            input_mask = np.array(input_mask)
            speaker_ids = np.array(speaker_ids)
            assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

            # Genre
            genre = genre_dict.get(doc_key[:2], 0)

            # Gold spans
            if len(gold_mentions) > 0:
                gold_starts, gold_ends = zip(*gold_mentions)
            else:
                gold_starts, gold_ends = [], []
            gold_starts = np.array(gold_starts)
            gold_ends = np.array(gold_ends)

            # Others
            tokens = json_data["tokens"]
            gold_clusters = json_data["clusters"]
            subtoken_map = json_data.get("subtoken_map", None)

            # DataInstanceに変換
            kargs = {
                "doc_key": doc_key,
                "tokens": tokens,
                "sentences": sentences,
                "sentence_map": sentence_map,
                "speakers": speakers,
                "gold_clusters": gold_clusters,
                "subtoken_map": subtoken_map,
                #
                "input_ids": input_ids,
                "input_mask": input_mask,
                "speaker_ids": speaker_ids,
                "sentence_len": sentence_len,
                "genre": genre,
                "sentence_map": sentence_map,
                "is_training": is_training,
                "gold_starts": gold_starts,
                "gold_ends": gold_ends,
                "gold_mention_cluster_map": gold_mention_cluster_map,
            }

            data = utils.DataInstance(**kargs)
            dataset.append(data)

    dataset = np.asarray(dataset, dtype="O")

    output_file = os.path.basename(input_file).replace(".jsonlines", ".npy")
    output_file = os.path.join(config["caches"], output_file)
    np.save(output_file, dataset)
    print("Cached %s to %s" % (input_file, output_file))


def get_speaker_dict(speakers, max_num_speakers):
    """
    Parameters
    ----------
    speakers: list[str]

    Returns
    -------
    dict[str, int]
    """
    speaker_dict = {"UNK": 0, "[SPL]": 1}
    for speaker in speakers:
        if len(speaker_dict) > max_num_speakers:
            pass # "break" to limit # speakers
        if speaker not in speaker_dict:
            speaker_dict[speaker] = len(speaker_dict)
    return speaker_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument("--is_training", type=int, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--seg_len', type=int, required=True)
    args = parser.parse_args()
    main(args)

