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
        elif self.config["model_name"] == "corefmodel4":
            self.model = models.CorefModel4(device=device,
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
        sentence_len = data.sentence_len
        genre = data.genre
        sentence_map = data.sentence_map
        is_training = data.is_training
        gold_starts = data.gold_starts
        gold_ends = data.gold_ends
        gold_mention_cluster_map = data.gold_mention_cluster_map

        if len(data.sentences) > self.config["truncation_size"]:
            input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
            is_training, gold_starts, gold_ends, gold_mention_cluster_map \
                = self.truncate_example(input_ids=input_ids,
                                        input_mask=input_mask,
                                        speaker_ids=speaker_ids,
                                        sentence_len=sentence_len,
                                        genre=genre,
                                        sentence_map=sentence_map,
                                        is_training=is_training,
                                        gold_starts=gold_starts,
                                        gold_ends=gold_ends,
                                        gold_mention_cluster_map=gold_mention_cluster_map)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_mask = torch.tensor(input_mask, dtype=torch.long, device=self.device)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long, device=self.device)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long, device=self.device)
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
                                     sentence_len=sentence_len,
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
        (list[(int, int)], list[list[(int, int)]], list[int])
        CorefEvaluator or None
        """
        if evaluator is None or gold_clusters is None:
            assert evaluator is None and gold_clusters is None

        # Tensorize inputs
        # data_gpu = [x.to(self.device) for x in data] # old code
        input_ids = torch.tensor(data.input_ids, dtype=torch.long, device=self.device)
        input_mask = torch.tensor(data.input_mask, dtype=torch.long, device=self.device)
        speaker_ids = torch.tensor(data.speaker_ids, dtype=torch.long, device=self.device)
        sentence_len = torch.tensor(data.sentence_len, dtype=torch.long, device=self.device)
        genre = torch.tensor(data.genre, dtype=torch.long, device=self.device)
        sentence_map = torch.tensor(data.sentence_map, dtype=torch.long, device=self.device)
        is_training = torch.tensor(data.is_training, dtype=torch.bool, device=self.device)
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
                                     sentence_len=sentence_len,
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

        # Get clusters
        predicted_mentions = [(s, e) for s, e in zip(span_starts, span_ends)]
        predicted_clusters, mention_to_cluster_id, antecedents = self.get_predicted_clusters(
                                                                span_starts=span_starts,
                                                                span_ends=span_ends,
                                                                antecedent_idx=antecedent_idx,
                                                                antecedent_scores=antecedent_scores)

        if evaluator is None:
            return [predicted_mentions, predicted_clusters, antecedents], None

        # Update evaluator
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        return [predicted_mentions, predicted_clusters, antecedents], evaluator

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores, require_indices=False):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        predicted_clusters = [] # list of list of (int, int)
        mention_to_cluster_id = {} # {(int, int): int}
        if require_indices:
            predicted_clusters_by_indices = [] # list of list of int
        for span_i, antecedent_i in enumerate(predicted_antecedents):
            # No coreference
            if antecedent_i < 0:
                continue

            # Check whether the coreference is valid
            assert antecedent_i < span_i, f'antecedent (index {antecedent_i}) must appear earlier than span (index {span_i})'

            # Add antecedent to cluster (if the antecedent is chosen for the first time)
            antecedent = (int(span_starts[antecedent_i]), int(span_ends[antecedent_i])) # Antecedent span
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1) # Cluster ID
            if antecedent_cluster_id == -1:
                # Add a new cluster
                antecedent_cluster_id = len(predicted_clusters) # New cluster ID
                predicted_clusters.append([antecedent]) # Add antecedent to cluster
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
                if require_indices:
                    predicted_clusters_by_indices.append([antecedent_i])

            # Add mention to cluster
            mention = (int(span_starts[span_i]), int(span_ends[span_i])) # Mention span
            predicted_clusters[antecedent_cluster_id].append(mention) # Add mention to cluster
            mention_to_cluster_id[mention] = antecedent_cluster_id
            if require_indices:
                predicted_clusters_by_indices[antecedent_cluster_id].append(span_i)

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        # predicted_clusters_by_indices = [tuple(c) for c in predicted_clusters_by_indices]
        if require_indices:
            return predicted_clusters, mention_to_cluster_id, predicted_antecedents, predicted_clusters_by_indices
        else:
            return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def truncate_example(self,
                         input_ids,
                         input_mask,
                         speaker_ids,
                         sentence_len,
                         genre,
                         sentence_map,
                         is_training,
                         gold_starts,
                         gold_ends,
                         gold_mention_cluster_map,
                         sentence_offset=None):
        truncation_size = self.config["truncation_size"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > truncation_size

        # Get offsets
        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - truncation_size) # Random!
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + truncation_size].sum()

        # Extract continuous segments
        input_ids = input_ids[sent_offset: sent_offset + truncation_size, :]
        input_mask = input_mask[sent_offset: sent_offset + truncation_size, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + truncation_size, :]
        sentence_len = sentence_len[sent_offset: sent_offset + truncation_size]
        sentence_map = sentence_map[word_offset: word_offset + num_words]

        # Get gold spans within the window
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset # Adjust token indices
        gold_ends = gold_ends[gold_spans] - word_offset # Adjust token indices
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]

        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map





