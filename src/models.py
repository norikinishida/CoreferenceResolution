from collections import Iterable

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel

import utils

import util


class Biaffine(nn.Module):

    def __init__(self, input_dim, output_dim=1, bias_x=True, bias_y=True):
        """
        Parameters
        ----------
        input_dim: int
        output_dim: int
        bias_x: bool
        bias_y: bool
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim+bias_x, input_dim+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        # nn.init.zeros_(self.weight)
        init.normal_(self.weight, std=0.02)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor(shape=(batch_size, seq_len, input_dim))
        y: torch.Tensor(shape=(batch_size, seq_len, input_dim))

        Returns
        -------
        torch.Tensor(shape=(batch_size, output_dim, seq_len, seq_len))
            A scoring tensor of shape ``[batch_size, output_dim, seq_len, seq_len]``.
            If ``output_dim=1``, the dimension for ``output_dim`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) # (batch_size, seq_len, input_dim+1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1) # (batch_size, seq_len, input_dim+1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y) # (batch_size, output_dim, seq_len, seq_len)

        return s


class Joshi2020(nn.Module):
    """
    c.f., https://github.com/lxucs/coref-hoi.git
    """
    def __init__(self, config, device, num_genres=None):
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.config = config
        self.device = device
        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']

        ########################
        # Model components
        ########################

        self.dropout = nn.Dropout(p=config['dropout_rate'])

        # For token embedding
        self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'], return_dict=False)
        self.bert_emb_dim = self.bert.config.hidden_size

        # For span embedding
        self.span_dim = self.bert_emb_dim * 2
        self.span_dim += config['feature_dim'] # span width
        if self.config["use_head_attn"]:
            self.span_dim += self.bert_emb_dim
        self.embed_span_width = self.make_embedding(dict_size=self.max_span_width, dim=config["feature_dim"])
        self.ffnn_mention_attn = self.make_ffnn(feat_dim=self.bert_emb_dim, hidden_dims=0, output_dim=1) if config["use_head_attn"] else None

        # For mention scoring
        self.ffnn_span_emb_score = self.make_ffnn(feat_dim=self.span_dim, hidden_dims=[config['ffnn_dim']] * config['ffnn_depth'], output_dim=1)

        # For mention scoring (prior)
        self.embed_span_width_prior = self.make_embedding(dict_size=self.max_span_width, dim=config["feature_dim"])
        self.ffnn_span_width_score = self.make_ffnn(feat_dim=config['feature_dim'], hidden_dims=[config['ffnn_dim']] * config['ffnn_depth'], output_dim=1)

        # For coreference scoring (coarse-grained)
        # self.coarse_bilinear = self.make_ffnn(feat_dim=self.span_dim, hidden_dims=0, output_dim=self.span_dim)
        self.biaffine_coref_score = Biaffine(input_dim=self.span_dim, output_dim=1, bias_x=False, bias_y=True)

        # For coreference scoring (prior)
        self.embed_antecedent_distance_prior = self.make_embedding(dict_size=10, dim=config["feature_dim"])
        self.ffnn_antecedent_distance_score = self.make_ffnn(feat_dim=config['feature_dim'], hidden_dims=0, output_dim=1)

        # For coreference scoring (fine-grained)
        self.pair_dim = self.span_dim * 3
        self.pair_dim += config['feature_dim'] # same-speaker indicator
        self.pair_dim += config['feature_dim'] # genre
        self.pair_dim += config['feature_dim'] # segment distance
        self.pair_dim += config['feature_dim'] # top antecedent distance
        self.embed_same_speaker = self.make_embedding(dict_size=2, dim=config["feature_dim"])
        self.embed_genre = self.make_embedding(dict_size=self.num_genres, dim=config["feature_dim"])
        self.embed_segment_distance = self.make_embedding(dict_size=config['max_training_sentences'], dim=config["feature_dim"])
        self.embed_top_antecedent_distance = self.make_embedding(dict_size=10, dim=config["feature_dim"])
        self.ffnn_coref_score = self.make_ffnn(feat_dim=self.pair_dim, hidden_dims=[config['ffnn_dim']] * config['ffnn_depth'], output_dim=1)

        ########################
        # Others
        ########################

        self.update_steps = 0  # Internal use for debug
        self.debug = True

    ########################
    # Component makers
    ########################

    def make_embedding(self, dict_size, dim, std=0.02):
        emb = nn.Embedding(dict_size, dim)
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_dim, hidden_dims, output_dim):
        if hidden_dims is None or hidden_dims == 0 or hidden_dims == [] or hidden_dims == [0]:
            return self.make_linear(feat_dim, output_dim)

        if not isinstance(hidden_dims, Iterable):
            hidden_dims = [hidden_dims]
        ffnn = [self.make_linear(feat_dim, hidden_dims[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_dims)):
            ffnn += [self.make_linear(hidden_dims[i-1], hidden_dims[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*ffnn)

    ################
    # For optimization
    ################

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    ################
    # Forwarding
    ################

    def forward(self,
                input_ids,
                input_mask,
                speaker_ids,
                sentence_len, # never used
                genre,
                sentence_map,
                is_training,  # never used
                gold_starts=None,
                gold_ends=None,
                gold_mention_cluster_map=None):
        """
        Parameters
        ----------
        input_ids: torch.tensor(shape=(n_segs, max_seg_len))
        input_mask: torch.Tensor(shape=(n_segs, max_seg_len))
        speaker_ids: torch.Tensor(shape=(n_segs, max_seg_len))
        sentence_len: torch.Tensor(shape=(n_segs,))
        genre: torch.Tensor of long
        sentence_map: torch.Tensor(shape=(n_tokens,))
        is_training: torch.Tensor of bool
        gold_starts: torch.Tensor(shape=(n_gold_spans,)), default None
        gold_ends: torch.Tensor(shape=(n_gold_spans,)), default None
        gold_mention_cluster_map: torch.Tensor(shape=(n_gold_spans,)), default None

        Returns
        (torch.Tensor(shape=(n_top_spans,)), torch.Tensor(shape=(n_top_spans,)), torch.Tensor(shape=(n_top_spans, n_ant_spans)), torch.Tensor(shape=(n_top_spans, n_ant_spans + 1)))
        torch.Tensor of float
        -------
        """
        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

        ###################################
        # <1> Token embeddings
        ###################################

        token_embs, _ = self.bert(input_ids, attention_mask=input_mask) # (n_segs, max_seg_len, tok_dim)
        input_mask = input_mask.to(torch.bool) # (n_segs, max_seg_len)
        token_embs = token_embs[input_mask] # (n_tokens, tok_dim)
        speaker_ids = speaker_ids[input_mask] # (n_tokens,)
        n_tokens = token_embs.shape[0]

        ###################################
        # <2> Candidate spans
        #
        # cand_span_start_token_indices =
        #   [[0, 0, 0, ...],
        #    [1, 1, 1, ...],
        #    [2, 2, 2, ...],
        #    ...
        #    [n_tokens-1, n_tokens-1, n_tokens-1, ...]]
        #
        # cand_span_end_token_indices =
        #   [[0, 1, 2, ...],
        #    [1, 2, 3, ...],
        #    [2, 3, 4, ...],
        #    ...
        #    [n_tokens-1, n_tokens, n_tokens+1, ...]]
        #
        ###################################

        sentence_indices = sentence_map  # (n_tokens,); sentence index of i-th token
        cand_span_start_token_indices = torch.unsqueeze(torch.arange(0, n_tokens, device=self.device), 1).repeat(1, self.max_span_width) # (n_tokens, max_span_width)
        cand_span_end_token_indices = cand_span_start_token_indices + torch.arange(0, self.max_span_width, device=self.device) # (n_tokens, max_span_width)
        cand_span_start_sent_indices = sentence_indices[cand_span_start_token_indices] # (n_tokens, max_span_width); sentence index
        cand_span_end_sent_indices = sentence_indices[torch.min(cand_span_end_token_indices, torch.tensor(n_tokens - 1, device=self.device))] # (n_tokens, max_span_width)
        cand_span_mask = (cand_span_end_token_indices < n_tokens) & (cand_span_start_sent_indices == cand_span_end_sent_indices)
        cand_span_start_token_indices = cand_span_start_token_indices[cand_span_mask]  # (n_cand_spans,)
        cand_span_end_token_indices = cand_span_end_token_indices[cand_span_mask]  # (n_cand_spans,)

        n_cand_spans = cand_span_start_token_indices.shape[0]

        if do_loss:
            same_starts = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(cand_span_start_token_indices, 0)) # (n_gold_spans, n_cand_spans)
            same_ends = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(cand_span_end_token_indices, 0)) # (n_gold_spans, n_cand_spans)
            same_spans = (same_starts & same_ends).to(torch.long) # (n_gold_spans, n_cand_spans)
            lhs = torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float) # (1, n_gold_spans)
            rhs = same_spans.to(torch.float) # (n_gold_spans, n_cand_spans)
            cand_span_cluster_ids = torch.matmul(lhs, rhs) # (1, n_cand_spans); cluster ids
            cand_span_cluster_ids = torch.squeeze(cand_span_cluster_ids.to(torch.long), 0) # (n_cand_spans,); non-gold span has cluster id 0

        ###################################
        # <3> Span embeddings
        ###################################

        # Get span endpoints embeddings
        cand_span_start_embs = token_embs[cand_span_start_token_indices] # (n_cand_spans, tok_dim)
        cand_span_end_embs = token_embs[cand_span_end_token_indices] # (n_cand_spans, tok_dim)
        cand_span_embs_list = [cand_span_start_embs, cand_span_end_embs]

        # Get span-width embedding
        cand_span_width_indices = cand_span_end_token_indices - cand_span_start_token_indices # (n_cand_spans,)
        cand_span_width_embs = self.embed_span_width(cand_span_width_indices) # (n_cand_spans, feat_dim)
        cand_span_width_embs = self.dropout(cand_span_width_embs)
        cand_span_embs_list.append(cand_span_width_embs)

        # Get span attention embedding
        if self.config["use_head_attn"]:
            token_attns = torch.squeeze(self.ffnn_mention_attn(token_embs), 1) # (n_tokens,)
            doc_range = torch.arange(0, n_tokens).to(self.device) # (n_tokens,)
            doc_range_1 = cand_span_start_token_indices.unsqueeze(1) <= doc_range # (n_cand_spans, n_tokens)
            doc_range_2 = doc_range <= cand_span_end_token_indices.unsqueeze(1) # (n_cand_spans, n_tokens)
            cand_span_token_mask = doc_range_1 & doc_range_2 # (n_cand_spans, n_tokens)
            cand_span_token_attns = torch.log(cand_span_token_mask.float()) + torch.unsqueeze(token_attns, 0) # (n_cand_spans, n_tokens); masking for spans (w/ broadcasting)
            cand_span_token_attns = nn.functional.softmax(cand_span_token_attns, dim=1) # (n_cand_spans, n_tokens)
            cand_span_ha_embs = torch.matmul(cand_span_token_attns, token_embs) # (n_cand_spans, tok_dim)
            cand_span_embs_list.append(cand_span_ha_embs)

        # concatenation
        cand_span_embs = torch.cat(cand_span_embs_list, dim=1)  # (n_cand_spans, span_dim)

        ###################################
        # <4> Mention scores
        ###################################

        # Get mention scores
        cand_span_mention_scores = torch.squeeze(self.ffnn_span_emb_score(cand_span_embs), 1) # (n_cand_spans,)

        # + Prior
        width_scores = torch.squeeze(self.ffnn_span_width_score(self.embed_span_width_prior.weight), 1) # (max_span_width,)
        cand_span_width_scores = width_scores[cand_span_width_indices] # (n_cand_spans,)
        cand_span_mention_scores = cand_span_mention_scores + cand_span_width_scores

        ###################################
        # <5> Top spans
        ###################################

        if do_loss:
            # We force the model to include the gold mentions in the top-ranked candidate spans during training
            cand_span_mention_scores_editted = torch.clone(cand_span_mention_scores)
            cand_span_mention_scores_editted[cand_span_cluster_ids > 0] = 100000.0 # Set large value to the gold mentions for trianing
            cand_span_indices_sorted_by_score = torch.argsort(cand_span_mention_scores_editted, descending=True).tolist() # (n_cand_spans,); candidate-span index
        else:
            cand_span_indices_sorted_by_score = torch.argsort(cand_span_mention_scores, descending=True).tolist() # (n_cand_spans,); candidate-span index
        n_top_spans = int(min(self.config['max_num_extracted_spans'], self.config['top_span_ratio'] * n_tokens))
        cand_span_start_token_indices_cpu = cand_span_start_token_indices.tolist() # (n_cand_spans,); token index
        cand_span_end_token_indices_cpu = cand_span_end_token_indices.tolist() # (n_cand_spans,); token index
        top_span_indices_cpu = self._extract_top_spans(cand_span_indices_sorted_by_score,
                                                       cand_span_start_token_indices_cpu,
                                                       cand_span_end_token_indices_cpu,
                                                       n_top_spans) # (n_top_spans,); candidate-span indices
        assert len(top_span_indices_cpu) == n_top_spans
        top_span_indices = torch.tensor(top_span_indices_cpu, device=self.device) # (n_top_spans,)

        top_span_start_token_indices = cand_span_start_token_indices[top_span_indices] # (n_top_spans,); token index
        top_span_end_token_indices = cand_span_end_token_indices[top_span_indices] # (n_top_spans,); token index
        top_span_embs = cand_span_embs[top_span_indices] # (n_top_spans, span_dim)
        top_span_cluster_ids = cand_span_cluster_ids[top_span_indices] if do_loss else None # (n_top_spans,); cluster ids
        top_span_mention_scores = cand_span_mention_scores[top_span_indices] # (n_top_spans,)

        ###################################
        # <6> Coarse-grained coreference scores
        ###################################
        """
        antecendet_offsets =
          [[0, -1, -2, -3, ..., -n_top_spans+1],
           [1,  0, -1, -2, ..., -n_top_spans+2],
           [2,  1,  0, -1, ..., -n_top_spans+3],
           ...
           [n_top_spans-1,  n_top_spans-2,  n_top_spans-3, -n_top_spans-4, ..., 0]]

        antecendet_mask =
          [[0, 0, 0, 0, ..., 0],
           [1, 0, 0, 0, ..., 0],
           [1, 1, 0, 0, ..., 0],
           ...
           [1, 1, 1, 1, ..., 0]]
        """

        # Get pairwise mention scores
        pairwise_mention_scores = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0) # (n_top_spans, n_top_spans)

        # Get pairwise coarse-grained coreference scores
        #
        # source_span_embs = self.dropout(self.coarse_bilinear(top_span_embs)) # (n_top_spans, span_dim)
        # target_span_embs = self.dropout(torch.transpose(top_span_embs, 0, 1)) # (span_dim, n_top_spans)
        # pairwise_coref_scores = torch.matmul(source_span_embs, target_span_embs) # (n_top_spans, n_top_spans)
        #
        source_span_embs = self.dropout(top_span_embs).unsqueeze(0) # (1, n_top_spans, span_dim)
        target_span_embs = self.dropout(top_span_embs).unsqueeze(0) # (1, n_top_spans, span_dim)
        pairwise_coref_scores = self.biaffine_coref_score(source_span_embs, target_span_embs).squeeze(0).squeeze(0) # (n_top_spans, n_top_spans)
        #
        pairwise_coref_scores = pairwise_mention_scores + pairwise_coref_scores # (n_top_spans, n_top_spans)

        # Masking
        top_span_range = torch.arange(0, n_top_spans, device=self.device) # (n_top_spans,); top-span index
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0) # (n_top_spans, n_top_spans)
        antecedent_mask = (antecedent_offsets >= 1) # (n_top_spans, n_top_spans); only preceding spans are active
        antecedent_mask = antecedent_mask.to(torch.float)
        pairwise_coref_scores = pairwise_coref_scores + torch.log(antecedent_mask) # (n_top_spans, n_top_spans)

        # + Prior
        distance_scores = torch.squeeze(self.ffnn_antecedent_distance_score(self.dropout(self.embed_antecedent_distance_prior.weight)), 1)
        bucketed_distances = util.bucket_distance(antecedent_offsets)
        antecedent_distance_scores = distance_scores[bucketed_distances]
        pairwise_coref_scores = pairwise_coref_scores + antecedent_distance_scores

        ###################################
        # <7> Antecedents (top-ranked mentions)
        ###################################

        #NOTE: antecedentsはcoarse-grained coreference scoresに基づいて選択される

        n_ant_spans = min(n_top_spans, self.config['max_top_antecedents'])
        top_antecedent_scores_coarse, top_antecedent_indices = torch.topk(pairwise_coref_scores, k=n_ant_spans) # (n_top_spans, n_ant_spans), (n_top_spans, n_ant_spans); j-th best top-span index for i-th top-span
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_indices, device=self.device) # (n_top_spans, n_ant_spans)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_indices, device=self.device)  # (n_top_spans, n_ant_spans)

        ###################################
        # <8> Fine-grained coreference scores
        ###################################

        # NOTE: lossはfine-grained coreference scoresに基づいて計算される

        if self.config['fine_grained']:

            ###################################
            # Get rich pair embeddings
            ###################################

            # Embeddings about whether two spans belong to the same speaker or not
            top_span_speaker_ids = speaker_ids[top_span_start_token_indices] # (n_top_spans,); speaker id of i-th span
            top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedent_indices] # (n_top_spans, n_ant_spans); speaker id of j-th antecedent of i-th span
            same_speakers = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_ids # (n_top_spans, n_ant_spans)
            same_speakers = same_speakers.to(torch.long)
            same_speaker_embs = self.embed_same_speaker(same_speakers) # (n_top_spans, n_ant_spans, feat_dim)

            # Embedding of the document's genre
            genre_embs = self.embed_genre(genre) # (feat_dim,)
            genre_embs = torch.unsqueeze(torch.unsqueeze(genre_embs, 0), 0).repeat(n_top_spans, n_ant_spans, 1) # (n_top_spans, n_ant_spans, feat_dim)

            # Embeddings of segment distance between spans
            n_segs, max_seg_len = input_ids.shape[0], input_ids.shape[1]
            token_seg_ids = torch.arange(0, n_segs, device=self.device).unsqueeze(1).repeat(1, max_seg_len) # (n_segs, max_seg_len); segment index of (i,j)-th token
            token_seg_ids = token_seg_ids[input_mask] # (seg_len,); segment index of i-th token
            top_span_seg_ids = token_seg_ids[top_span_start_token_indices] # (n_top_spans,); segment index of i-th span
            top_antecedent_seg_ids = token_seg_ids[top_span_start_token_indices[top_antecedent_indices]] # (n_top_spans, n_ant_spans); segment index of j-th antecedent of i-th span
            top_antecedent_seg_distances = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids # (n_top_spans, n_ant_spans); segment-level distance between i-th span and j-th span
            top_antecedent_seg_distances = torch.clamp(top_antecedent_seg_distances, 0, self.config['max_training_sentences'] - 1) # (n_top_spans, n_ant_spans)
            seg_distance_embs = self.embed_segment_distance(top_antecedent_seg_distances) # (n_top_spans, n_ant_spans, feat_dim)

            # Embeddings of index-level distance between spans
            # Is index-level distance meaningful???
            top_antecedent_distances = util.bucket_distance(top_antecedent_offsets) # (n_top_spans, n_ant_spans); span-index-level distance
            top_antecedent_distance_embs = self.embed_top_antecedent_distance(top_antecedent_distances) # (n_top_spans, n_ant_spans, feat_dim)

            # Concat
            feature_list = [same_speaker_embs, genre_embs, seg_distance_embs, top_antecedent_distance_embs]
            feature_embs = torch.cat(feature_list, dim=2) # (n_top_spans, n_ant_spans, 4 * feat_dim)
            feature_embs = self.dropout(feature_embs) # (n_top_spans, n_ant_spans, 4 * feat_dim)

            # top spans (lhs) x antecedent spans (rhs) x their product
            lhs_embs = torch.unsqueeze(top_span_embs, 1).repeat(1, n_ant_spans, 1) # (n_top_spans, n_ant_spans, span_dim)
            rhs_embs = top_span_embs[top_antecedent_indices]  # (n_top_spans, n_ant_spans, span_dim)
            product_embs = lhs_embs * rhs_embs # (n_top_spans, n_ant_spans, span_dim)

            # Concat
            pair_embs = torch.cat([lhs_embs, rhs_embs, product_embs, feature_embs], 2) # (n_top_spans, n_ant_spans, 3*span_dim + 4*feat_dim)

            ###################################
            # Get fine-grained coreference scores
            ###################################

            top_antecedent_scores_fine = torch.squeeze(self.ffnn_coref_score(pair_embs), 2) # (n_top_spans, n_ant_spans)
            top_antecedent_scores = top_antecedent_scores_coarse + top_antecedent_scores_fine # (n_top_spans, n_ant_spans)

        else:
            top_antecedent_scores = top_antecedent_scores_coarse # (n_top_spans, n_ant_spans)

        top_antecedent_scores = torch.cat([torch.zeros(n_top_spans, 1, device=self.device), top_antecedent_scores], dim=1) # (n_top_spans, 1+n_ant_spans)

        if not do_loss:
            return [top_span_start_token_indices, \
                    top_span_end_token_indices, \
                    top_antecedent_indices, \
                    top_antecedent_scores], None

        ###################################
        # <9> Loss
        ###################################

        # A integer matrix, where (i,j)-th element is the cluster id of i-th span and j-th antecedent
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_indices] # (n_top_spans, n_ant_spans)
        # A integer matrix, where (i,j)-th element is negative if the antecedent is invalid
        top_antecedent_cluster_ids = top_antecedent_cluster_ids + (top_antecedent_mask.to(torch.long) - 1) * 100000
        # A bool matrix, where (i,j)-th element indicates whether j-th antecedent is valid and belongs to the same cluster with i-th span (True), or not (False)
        gold_cluster_indicators = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1)) # (n_top_spans, n_ant_spans)
        # A bool vector, where i-th element indicates whether i-th span is valid (True) or not (False)
        non_dummy_indicators = torch.unsqueeze(top_span_cluster_ids > 0, 1) # (n_top_spans, 1)
        # A bool matrix, where rows of invalid spans are masked with False
        masked_gold_cluster_indicators = gold_cluster_indicators & non_dummy_indicators # (n_top_spans, n_ant_spans)
        # A bool vector, where i-th element indicates whether the span is "invalid" (True) or not (False)
        dummy_cluster_indicators = torch.logical_not(masked_gold_cluster_indicators.any(dim=1, keepdims=True)) # (n_top_spans, 1)
        n_singleton_or_invalid_top_spans = float(dummy_cluster_indicators.to(torch.long).sum())
        # A binary matrix, where (i,j)-th element indicates whether the antecedent is valid and belongs to the same cluster with i-th span (1) or not (0)
        complete_gold_cluster_indicators = torch.cat([dummy_cluster_indicators, masked_gold_cluster_indicators], dim=1) # (n_top_spans, 1+n_ant_spans)
        complete_gold_cluster_indicators = complete_gold_cluster_indicators.to(torch.float)

        # Main loss
        # Loss = sum_{i} L_{i}
        # L_{i}
        #   = -log[ sum_{k} exp(y_{i,k} + m_{i,k}) / sum_{k} exp(y_{i,k}) ]
        #   = -( log[ sum_{k} exp(y_{i,k} + m_{i,k}) ] - log[ sum_{k} exp(y_{i,k}) ] )
        #   = log[ sum_{k} exp(y_{i,k}) ] - log[ sum_{k} exp(y_{i,k} + m_{i,k}) ]
        log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
        log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + torch.log(complete_gold_cluster_indicators), dim=1)
        loss = torch.sum(log_norm - log_marginalized_antecedent_scores)

        # Add mention loss
        if self.config['mention_loss_coef'] > 0:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            #
            # gold_mention_scores = cand_span_mention_scores[cand_span_cluster_ids > 0]
            # non_gold_mention_scores = cand_span_mention_scores[cand_span_cluster_ids == 0]
            #
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * self.config['mention_loss_coef']
            loss_mention = loss_mention - torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * self.config['mention_loss_coef']
            loss = loss + loss_mention

        ###################################
        # Others
        ###################################

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                utils.writelog('---------debug step: %d---------' % self.update_steps)
                utils.writelog(f"segments: {input_ids.shape[0]}; tokens: {n_tokens}; candidate spans: {n_cand_spans}; top spans: {n_top_spans}")
                utils.writelog("gold mentions: %d -> %d (recall: %.2f) -> %d (recall: %.2f, ratio: %.2f); gold coreferents: %d (ratio: %.2f)" % \
                                (gold_starts.shape[0],
                                 (cand_span_cluster_ids > 0).sum(),
                                 (cand_span_cluster_ids > 0).sum()/gold_starts.shape[0],
                                 (top_span_cluster_ids > 0).sum(),
                                 (top_span_cluster_ids > 0).sum()/gold_starts.shape[0],
                                 (top_span_cluster_ids > 0).sum()/n_top_spans,
                                 n_top_spans - n_singleton_or_invalid_top_spans,
                                 (n_top_spans - n_singleton_or_invalid_top_spans)/n_top_spans,
                                )
                              )
                utils.writelog('logsumexp(gold)/logsumexp(all): %.4f/%.4f' % (torch.sum(log_marginalized_antecedent_scores), torch.sum(log_norm)))
                if self.config['mention_loss_coef'] > 0:
                    utils.writelog('mention loss: %.4f' % loss_mention)
                utils.writelog('loss: %.4f' % loss)

        self.update_steps += 1

        return [top_span_start_token_indices,
                top_span_end_token_indices,
                top_antecedent_indices,
                top_antecedent_scores], loss

    def _extract_top_spans(self, cand_span_indices_sorted, cand_span_start_token_indices, cand_span_end_token_indices, n_top_spans):
        """
        Parameters
        ----------
        cand_span_indices_sorted: torch.Tensor(shape=(n_cand_spans,)); candidate-span index
        cand_span_start_token_indices: list[int]
        cand_span_end_token_indices: list[int]
        n_top_spans: int

        Returns
        -------
        list[int]
        """
        selected_cand_span_indices = []
        start_to_max_end, end_to_min_start = {}, {}
        for cand_span_idx in cand_span_indices_sorted:
            if len(selected_cand_span_indices) >= n_top_spans:
                break
            # Perform overlapping check
            span_start_token_idx = cand_span_start_token_indices[cand_span_idx]
            span_end_token_idx = cand_span_end_token_indices[cand_span_idx]
            cross_overlap = False
            for token_idx in range(span_start_token_idx, span_end_token_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_token_idx and max_end > span_end_token_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_token_idx and 0 <= min_start < span_start_token_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_cand_span_indices.append(cand_span_idx) # NOTE: cand_span_idx is already sorted in descending order
                max_end = start_to_max_end.get(span_start_token_idx, -1)
                if span_end_token_idx > max_end:
                    start_to_max_end[span_start_token_idx] = span_end_token_idx
                min_start = end_to_min_start.get(span_end_token_idx, -1)
                if min_start == -1 or span_start_token_idx < min_start:
                    end_to_min_start[span_end_token_idx] = span_start_token_idx
        # Sort selected candidates by span idx
        selected_cand_span_indices = sorted(selected_cand_span_indices, key=lambda idx: (cand_span_start_token_indices[idx], cand_span_end_token_indices[idx]))
        if len(selected_cand_span_indices) < n_top_spans:  # Padding
            selected_cand_span_indices += ([selected_cand_span_indices[0]] * (n_top_spans - len(selected_cand_span_indices)))
        return selected_cand_span_indices





