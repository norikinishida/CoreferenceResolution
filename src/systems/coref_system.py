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
            self.model = models.Joshi2020(device=device, config=config)
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
        # Switch to training mode
        self.model.train()

        # Forward
        _, loss = self.model.forward(data=data, get_loss=True)

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

        # Switch to inference mode
        self.model.eval()

        # Forward
        (span_starts, span_ends, antecedent_indices, antecedent_scores), _ \
                = self.model.forward(data=data, get_loss=False)

        span_starts = span_starts.tolist()
        span_ends = span_ends.tolist()
        antecedent_indices = antecedent_indices.tolist()
        antecedent_scores = antecedent_scores.tolist()

        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_indices=antecedent_indices, antecedent_scores=antecedent_scores)

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

    def get_predicted_antecedents(self, antecedent_indices, antecedent_scores):
        """
        Parameters
        ----------
        antecedent_indices: list[list[int]]
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
                predicted_antecedents.append(antecedent_indices[i][idx])
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
        mention_to_cluster_id = {} # dict[(int, int), int]
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

