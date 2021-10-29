import random

import numpy as np
import torch

import utils

import models


class CorefSystem:

    def __init__(self, device, config):
        """
        Parameters
        ----------
        device: str
        config: utils.Config
        """
        self.device = device
        self.config = config

        # Initialize model
        if self.config["model_name"] == "joshi2020":
            self.model = models.Joshi2020(device=device,
                                          config=config)
        else:
            raise Exception("Invalid model_name %s" % self.config["model_name"])

        # Show parameter shapes
        utils.writelog("Model parameters:")
        for name, param in self.model.named_parameters():
            utils.writelog("%s: %s" % (name, tuple(param.shape)))

    def load_model(self, path):
        """
        Parameters
        ----------
        path: str
        """
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")), strict=False)

    def save_model(self, path):
        """
        Parameters
        ----------
        path: str
        """
        torch.save(self.model.state_dict(), path)

    def to_gpu(self, device):
        """
        Parameters
        ----------
        device: str
        """
        self.model.to(device)

    def compute_loss(self, data):
        """
        Parameters
        ----------
        data: utils.DataInstance

        Returns
        -------
        torch.Tensor
        """
        # Tensorize inputs
        # data_gpu = [x.to(self.device) for x in data] # old code
        input_ids = data.input_ids
        input_mask = data.input_mask
        speaker_ids = data.speaker_ids
        segment_len = data.segment_len
        genre = data.genre
        sentence_map = data.sentence_map
        is_training = data.is_training
        gold_starts = data.gold_starts
        gold_ends = data.gold_ends
        gold_mention_cluster_map = data.gold_mention_cluster_map

        if len(data.segments) > self.config["truncation_size"]:
            input_ids, input_mask, speaker_ids, segment_len, genre, sentence_map, \
            is_training, gold_starts, gold_ends, gold_mention_cluster_map \
                = self.truncate_example(input_ids=input_ids,
                                        input_mask=input_mask,
                                        speaker_ids=speaker_ids,
                                        segment_len=segment_len,
                                        genre=genre,
                                        sentence_map=sentence_map,
                                        is_training=is_training,
                                        gold_starts=gold_starts,
                                        gold_ends=gold_ends,
                                        gold_mention_cluster_map=gold_mention_cluster_map)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_mask = torch.tensor(input_mask, dtype=torch.long, device=self.device)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long, device=self.device)
        segment_len = torch.tensor(segment_len, dtype=torch.long, device=self.device)
        genre = torch.tensor(genre, dtype=torch.long, device=self.device)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long, device=self.device)
        is_training = torch.tensor(is_training, dtype=torch.bool, device=self.device)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long, device=self.device)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long, device=self.device)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long, device=self.device)

        # Switch to training mode
        self.model.train()

        # Forward
        _, loss = self.model.forward(input_ids=input_ids,
                                     input_mask=input_mask,
                                     speaker_ids=speaker_ids,
                                     segment_len=segment_len,
                                     genre=genre,
                                     sentence_map=sentence_map,
                                     is_training=is_training,
                                     gold_starts=gold_starts,
                                     gold_ends=gold_ends,
                                     gold_mention_cluster_map=gold_mention_cluster_map)

        return loss

    def predict(self, data, evaluator=None, gold_clusters=None):
        """
        Parameters
        ----------
        data: utils.DataInstance
        evaluator: CorefEvaluator, default None
        gold_clusters: list[list[(int, int)]], default None

        Returns
        -------
        list[list[(int, int)]]
        CorefEvaluator or None
        """
        if evaluator is None or gold_clusters is None:
            assert evaluator is None and gold_clusters is None

        # Tensorize inputs
        input_ids = torch.tensor(data.input_ids, dtype=torch.long, device=self.device)
        input_mask = torch.tensor(data.input_mask, dtype=torch.long, device=self.device)
        speaker_ids = torch.tensor(data.speaker_ids, dtype=torch.long, device=self.device)
        segment_len = torch.tensor(data.segment_len, dtype=torch.long, device=self.device)
        genre = torch.tensor(data.genre, dtype=torch.long, device=self.device)
        sentence_map = torch.tensor(data.sentence_map, dtype=torch.long, device=self.device)
        is_training = torch.tensor(data.is_training, dtype=torch.bool, device=self.device)

        # Tensorize targets
        # gold_starts = torch.tensor(data.gold_starts, dtype=torch.long, device=self.device)
        # gold_ends = torch.tensor(data.gold_ends, dtype=torch.long, device=self.device)
        # gold_mention_cluster_map = torch.tensor(data.gold_mention_cluster_map, dtype=torch.long, device=self.device)

        # Switch to inference mode
        self.model.eval()

        # Forward
        (span_starts, span_ends, antecedent_idx, antecedent_scores), _ \
                    = self.model.forward(input_ids=input_ids,
                                         input_mask=input_mask,
                                         speaker_ids=speaker_ids,
                                         segment_len=segment_len,
                                         genre=genre,
                                         sentence_map=sentence_map,
                                         is_training=is_training,
                                         gold_starts=None,
                                         gold_ends=None,
                                         gold_mention_cluster_map=None)

        span_starts = span_starts.tolist()
        span_ends = span_ends.tolist()
        antecedent_idx = antecedent_idx.tolist()
        antecedent_scores = antecedent_scores.tolist()

        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx=antecedent_idx, antecedent_scores=antecedent_scores)

        # Get clusters
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(
                                                            span_starts=span_starts,
                                                            span_ends=span_ends,
                                                            predicted_antecedents=predicted_antecedents)

        if evaluator is None:
            return predicted_clusters, None

        # Update evaluator
        # mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        return predicted_clusters, evaluator

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """
        Parameters
        ----------
        antecedent_idx: list[list[int]]
            shape (n_top_spans, n_ant_spans)
        antecedent_scores: list[list[float]]
            shape (n_top_spans, 1 + n_ant_spans)

        Returns
        -------
        list[int]
        """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                # The dummy antecedent is selected.
                # Since the coreference score to the dummy antecedent is always set to zero,
                # the coreference scores to the non-dummy candidates are all negative.
                predicted_antecedents.append(-1)
            else:
                # The maximum antecedent score is positive,
                # and the selected antecedent is not dummy.
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, predicted_antecedents):
        """
        Parameters
        ----------
        span_starts: list[int]
        span_ends: list[int]
        predicted_antecedents: list[int]

        Returns
        -------
        list[list[(int, int)]]
        dict[(int, int), int]
        """
        # Get predicted clusters
        predicted_clusters = [] # list[list[(int, int)]]
        mention_to_cluster_id = {} # dict[(int, int), list[(int, int)]]
        for mention_i, antecedent_i in enumerate(predicted_antecedents):
            # No coreference
            if antecedent_i < 0:
                continue

            # Check whether the coreference is valid
            assert antecedent_i < mention_i, f'antecedent (index {antecedent_i}) must appear earlier than span (index {mention_i})'

            # Add antecedent to cluster (if the antecedent is chosen for the first time)
            antecedent = (int(span_starts[antecedent_i]), int(span_ends[antecedent_i])) # Antecedent span
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1) # Cluster ID
            if antecedent_cluster_id == -1:
                # Add a new cluster
                antecedent_cluster_id = len(predicted_clusters) # New cluster ID
                predicted_clusters.append([antecedent]) # Add antecedent to cluster
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            else:
                # This (antecedent) span is already selected as an antecedent of the previous mention(s)
                pass

            # Add mention to cluster
            mention = (int(span_starts[mention_i]), int(span_ends[mention_i])) # Mention span
            assert not mention in predicted_clusters[antecedent_cluster_id]
            assert not mention in mention_to_cluster_id
            predicted_clusters[antecedent_cluster_id].append(mention) # Add mention to cluster
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        return predicted_clusters, mention_to_predicted

    def truncate_example(self,
                         input_ids,
                         input_mask,
                         speaker_ids,
                         segment_len,
                         genre,
                         sentence_map,
                         is_training,
                         gold_starts,
                         gold_ends,
                         gold_mention_cluster_map,
                         segment_offset=None):
        truncation_size = self.config["truncation_size"]
        num_segments = input_ids.shape[0]
        assert num_segments > truncation_size

        # Get offsets
        if segment_offset is None:
            segment_offset = random.randint(0, num_segments - truncation_size) # Random!
        word_offset = segment_len[:segment_offset].sum()
        num_words = segment_len[segment_offset: segment_offset + truncation_size].sum()

        # Extract continuous segments
        input_ids = input_ids[segment_offset: segment_offset + truncation_size, :]
        input_mask = input_mask[segment_offset: segment_offset + truncation_size, :]
        speaker_ids = speaker_ids[segment_offset: segment_offset + truncation_size, :]
        segment_len = segment_len[segment_offset: segment_offset + truncation_size]
        sentence_map = sentence_map[word_offset: word_offset + num_words]

        # Get gold spans within the window
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset # Adjust token indices
        gold_ends = gold_ends[gold_spans] - word_offset # Adjust token indices
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]

        return input_ids, input_mask, speaker_ids, segment_len, genre, sentence_map, \
                is_training, gold_starts, gold_ends, gold_mention_cluster_map

