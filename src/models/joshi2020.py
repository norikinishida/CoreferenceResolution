import random

import torch
import torch.nn as nn
from transformers import AutoModel

import utils

import util
from . import shared_functions


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
        self.use_head_attn = config["use_head_attn"]
        self.use_features = config["use_features"]
        self.span_distance_type = config["span_distance_type"] # "segment" or "sentence"

        ########################
        # Model components
        ########################

        self.dropout = nn.Dropout(p=config['dropout_rate'])

        # For token embedding
        self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'], return_dict=False)
        self.bert_emb_dim = self.bert.config.hidden_size

        # For span embedding
        self.span_dim = self.bert_emb_dim * 2
        if self.use_features:
            self.span_dim += config['feature_dim'] # span width
        if self.use_head_attn:
            self.span_dim += self.bert_emb_dim

        if self.use_features:
            self.embed_span_width = shared_functions.make_embedding(dict_size=self.max_span_width, dim=config["feature_dim"])
        if self.use_head_attn:
            self.mlp_mention_attn = shared_functions.make_mlp(input_dim=self.bert_emb_dim, hidden_dims=0, output_dim=1, dropout=self.dropout)

        # For mention scoring
        self.mlp_span_emb_score = shared_functions.make_mlp(input_dim=self.span_dim, hidden_dims=[config['mlp_dim']] * config['mlp_depth'], output_dim=1, dropout=self.dropout)

        # For mention scoring (prior)
        if self.use_features:
            self.embed_span_width_prior = shared_functions.make_embedding(dict_size=self.max_span_width, dim=config["feature_dim"])
            self.mlp_span_width_score = shared_functions.make_mlp(input_dim=config['feature_dim'], hidden_dims=[config['mlp_dim']] * config['mlp_depth'], output_dim=1, dropout=self.dropout)

        # For coreference scoring (coarse-grained)
        # self.coarse_bilinear = self.make_mlp(input_dim=self.span_dim, hidden_dims=0, output_dim=self.span_dim)
        self.biaffine_coref_score = shared_functions.Biaffine(input_dim=self.span_dim, output_dim=1, bias_x=False, bias_y=True)

        # For coreference scoring (prior)
        if self.use_features:
            self.embed_antecedent_distance_prior = shared_functions.make_embedding(dict_size=10, dim=config["feature_dim"])
            self.mlp_antecedent_distance_score = shared_functions.make_mlp(input_dim=config['feature_dim'], hidden_dims=0, output_dim=1, dropout=self.dropout)

        # For coreference scoring (fine-grained)
        self.pair_dim = self.span_dim * 3 # mention, antecedent, product
        if self.use_features:
            self.pair_dim += config['feature_dim'] # same-speaker indicator
            self.pair_dim += config['feature_dim'] # genre
            self.pair_dim += config['feature_dim'] # segment distance
            self.pair_dim += config['feature_dim'] # top antecedent distance

            self.embed_same_speaker = shared_functions.make_embedding(dict_size=2, dim=config["feature_dim"])
            self.embed_genre = shared_functions.make_embedding(dict_size=self.num_genres, dim=config["feature_dim"])
            self.embed_segment_distance = shared_functions.make_embedding(dict_size=config['max_training_segments'], dim=config["feature_dim"])
            self.embed_top_antecedent_distance = shared_functions.make_embedding(dict_size=10, dim=config["feature_dim"])

        self.mlp_coref_score = shared_functions.make_mlp(input_dim=self.pair_dim, hidden_dims=[config['mlp_dim']] * config['mlp_depth'], output_dim=1, dropout=self.dropout)

        ########################
        # Others
        ########################

        self.update_steps = 0  # Internal use for debug
        self.debug = True

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

    def forward(self, data, get_loss):
        """
        Parameters
        ----------
        data: DataInstance
        get_loss: bool

        Returns
        (torch.Tensor(shape=(n_top_spans,)), torch.Tensor(shape=(n_top_spans,)), torch.Tensor(shape=(n_top_spans, n_ant_spans)), torch.Tensor(shape=(n_top_spans, n_ant_spans + 1)))
        torch.Tensor of float
        -------
        """
        ###################################
        # <0> Tensorize inputs (and targets)
        ###################################

        input_ids = data.input_ids # (n_segs, max_seg_len)
        input_mask = data.input_mask # (n_segs, max_seg_len)
        speaker_ids = data.speaker_ids # (n_segs, max_seg_len)
        segment_len = data.segment_len # (n_segs,)
        genre = data.genre # int
        sentence_map = data.sentence_map # (n_tokens,)
        is_training = data.is_training # bool
        if get_loss:
            gold_starts = data.gold_starts # (n_gold_spans,)
            gold_ends = data.gold_ends # (n_gold_spans,)
            gold_mention_cluster_map = data.gold_mention_cluster_map # (n_gold_spans,)

        if get_loss and len(data.segments) > self.config["truncation_size"]:
            input_ids,\
                input_mask,\
                speaker_ids,\
                segment_len,\
                genre,\
                sentence_map, \
                is_training,\
                gold_starts,\
                gold_ends,\
                gold_mention_cluster_map \
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
        if get_loss:
            gold_starts = torch.tensor(gold_starts, dtype=torch.long, device=self.device)
            gold_ends = torch.tensor(gold_ends, dtype=torch.long, device=self.device)
            gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long, device=self.device)

        ###################################
        # <1> Embed tokens using BERT
        ###################################

        token_embs, _ = self.bert(input_ids, attention_mask=input_mask) # (n_segs, max_seg_len, tok_dim)
        input_mask = input_mask.to(torch.bool) # (n_segs, max_seg_len)
        token_embs = token_embs[input_mask] # (n_tokens, tok_dim)
        speaker_ids = speaker_ids[input_mask] # (n_tokens,)
        n_tokens = token_embs.shape[0]

        ###################################
        # <2> Generate candidate spans with simple constraints
        ###################################
        """
        cand_span_start_token_indices =
          [[0, 0, 0, ...],
           [1, 1, 1, ...],
           [2, 2, 2, ...],
           ...
           [n_tokens-1, n_tokens-1, n_tokens-1, ...]]

        cand_span_end_token_indices =
          [[0, 1, 2, ...],
           [1, 2, 3, ...],
           [2, 3, 4, ...],
           ...
           [n_tokens-1, n_tokens, n_tokens+1, ...]]
        """

        cand_span_start_token_indices = torch.unsqueeze(torch.arange(0, n_tokens, device=self.device), 1).repeat(1, self.max_span_width) # (n_tokens, max_span_width)
        cand_span_end_token_indices = cand_span_start_token_indices + torch.arange(0, self.max_span_width, device=self.device) # (n_tokens, max_span_width)

        # sentence index of each token
        sentence_indices = sentence_map  # (n_tokens,); sentence index of i-th token
        clamped_cand_span_end_token_indices = torch.min(cand_span_end_token_indices, torch.tensor(n_tokens - 1, device=self.device))
        cand_span_start_sent_indices = sentence_indices[cand_span_start_token_indices] # (n_tokens, max_span_width)
        cand_span_end_sent_indices = sentence_indices[clamped_cand_span_end_token_indices] # (n_tokens, max_span_width)

        # condition: within document boundary & within same sentence
        cand_span_mask = cand_span_end_token_indices < n_tokens
        cand_span_mask = cand_span_mask & (cand_span_start_sent_indices == cand_span_end_sent_indices)
        cand_span_start_token_indices = cand_span_start_token_indices[cand_span_mask]  # (n_cand_spans,)
        cand_span_end_token_indices = cand_span_end_token_indices[cand_span_mask]  # (n_cand_spans,)

        n_cand_spans = cand_span_start_token_indices.shape[0]

        if get_loss:
            same_starts = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(cand_span_start_token_indices, 0)) # (n_gold_spans, n_cand_spans)
            same_ends = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(cand_span_end_token_indices, 0)) # (n_gold_spans, n_cand_spans)
            same_spans = (same_starts & same_ends).to(torch.long) # (n_gold_spans, n_cand_spans)
            lhs = torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float) # (1, n_gold_spans)
            rhs = same_spans.to(torch.float) # (n_gold_spans, n_cand_spans)
            cand_span_cluster_ids = torch.matmul(lhs, rhs) # (1, n_cand_spans); cluster ids
            cand_span_cluster_ids = torch.squeeze(cand_span_cluster_ids.to(torch.long), 0) # (n_cand_spans,); non-gold span has cluster id 0

        ###################################
        # <3> Compute span vectors for each candidate span
        ###################################

        # Get span endpoints embeddings
        cand_span_start_embs = token_embs[cand_span_start_token_indices] # (n_cand_spans, tok_dim)
        cand_span_end_embs = token_embs[cand_span_end_token_indices] # (n_cand_spans, tok_dim)
        cand_span_embs_list = [cand_span_start_embs, cand_span_end_embs]

        # Get span-width embedding
        if self.use_features:
            cand_span_width_indices = cand_span_end_token_indices - cand_span_start_token_indices # (n_cand_spans,)
            cand_span_width_embs = self.embed_span_width(cand_span_width_indices) # (n_cand_spans, feat_dim)
            cand_span_width_embs = self.dropout(cand_span_width_embs)
            cand_span_embs_list.append(cand_span_width_embs)

        # Get span attention embedding
        if self.config["use_head_attn"]:
            token_attns = torch.squeeze(self.mlp_mention_attn(token_embs), 1) # (n_tokens,)
            doc_range = torch.arange(0, n_tokens).to(self.device) # (n_tokens,)
            doc_range_1 = cand_span_start_token_indices.unsqueeze(1) <= doc_range # (n_cand_spans, n_tokens)
            doc_range_2 = doc_range <= cand_span_end_token_indices.unsqueeze(1) # (n_cand_spans, n_tokens)
            cand_span_token_mask = doc_range_1 & doc_range_2 # (n_cand_spans, n_tokens)
            if get_loss:
                cand_span_token_attns = torch.log(cand_span_token_mask.float()) + torch.unsqueeze(token_attns, 0) # (n_cand_spans, n_tokens); masking for spans (w/ broadcasting)
                cand_span_token_attns = nn.functional.softmax(cand_span_token_attns, dim=1) # (n_cand_spans, n_tokens)
                cand_span_ha_embs = torch.matmul(cand_span_token_attns, token_embs) # (n_cand_spans, tok_dim)
            else:
                cand_span_ha_embs_list = []
                for cand_i in range(cand_span_token_mask.shape[0]):
                    cropped_span_attns = token_attns[cand_span_token_mask[cand_i]] # (span_len,)
                    cropped_span_attns = nn.functional.softmax(cropped_span_attns) # (span_len,)
                    cropped_token_embs = token_embs[cand_span_token_mask[cand_i]] # (span_len, tok_dim)
                    cropped_span_ha_embs = torch.matmul(cropped_span_attns.unsqueeze(0), cropped_token_embs) # (1, tok_dim)
                    cand_span_ha_embs_list.append(cropped_span_ha_embs)
                cand_span_ha_embs = torch.cat(cand_span_ha_embs_list, axis=0) # (n_cand_spans, tok_dim)

            cand_span_embs_list.append(cand_span_ha_embs)

        # Concatenate
        cand_span_embs = torch.cat(cand_span_embs_list, dim=1)  # (n_cand_spans, span_dim)

        ###################################
        # <4> Compute mention scores for each candidate span
        ###################################

        # Get mention scores
        cand_span_mention_scores = torch.squeeze(self.mlp_span_emb_score(cand_span_embs), 1) # (n_cand_spans,)

        # + Prior (span width)
        if self.use_features:
            width_scores = torch.squeeze(self.mlp_span_width_score(self.embed_span_width_prior.weight), 1) # (max_span_width,)
            cand_span_width_scores = width_scores[cand_span_width_indices] # (n_cand_spans,)
            cand_span_mention_scores = cand_span_mention_scores + cand_span_width_scores

        ###################################
        # <5> Prune the candidate spans based on the mention scores
        ###################################

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
        if get_loss:
            top_span_cluster_ids = cand_span_cluster_ids[top_span_indices] # (n_top_spans,); cluster ids
        top_span_mention_scores = cand_span_mention_scores[top_span_indices] # (n_top_spans,)

        ###################################
        # <6> Compute coarse-grained coreference scores for each pair of top-ranked spans
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

        # Apply mask to invalid antecedents
        top_span_range = torch.arange(0, n_top_spans, device=self.device) # (n_top_spans,); top-span index
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0) # (n_top_spans, n_top_spans)
        antecedent_mask = (antecedent_offsets >= 1) # (n_top_spans, n_top_spans); only preceding spans are active
        antecedent_mask = antecedent_mask.to(torch.float)
        pairwise_coref_scores = pairwise_coref_scores + torch.log(antecedent_mask) # (n_top_spans, n_top_spans)

        # + Prior (antecedent distance)
        if self.use_features:
            distance_scores = torch.squeeze(self.mlp_antecedent_distance_score(self.dropout(self.embed_antecedent_distance_prior.weight)), 1)
            bucketed_distances = util.bucket_distance(antecedent_offsets)
            antecedent_distance_scores = distance_scores[bucketed_distances]
            pairwise_coref_scores = pairwise_coref_scores + antecedent_distance_scores

        ###################################
        # <7> Select top-K-ranked antecedents for each top-ranked span
        ###################################

        # NOTE: Antecedents are selected based on coarse-grained coreference scores

        n_ant_spans = min(n_top_spans, self.config['max_top_antecedents'])
        top_antecedent_scores_coarse, top_antecedent_indices = torch.topk(pairwise_coref_scores, k=n_ant_spans) # (n_top_spans, n_ant_spans), (n_top_spans, n_ant_spans); j-th best top-span index for i-th top-span
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_indices, device=self.device) # (n_top_spans, n_ant_spans)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_indices, device=self.device)  # (n_top_spans, n_ant_spans)

        ###################################
        # <8> Compute fine-grained coreference scores for the top-ranked coreference pairs
        ###################################

        # NOTE: Loss is computed based on fine-grained coreference scores

        if self.config['fine_grained']:

            ###################################
            # Get rich pair embeddings
            ###################################

            if self.use_features:
                # Embed the same-speaker indicators
                top_span_speaker_ids = speaker_ids[top_span_start_token_indices] # (n_top_spans,); speaker id of i-th span
                top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedent_indices] # (n_top_spans, n_ant_spans); speaker id of j-th antecedent of i-th span
                same_speakers = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_ids # (n_top_spans, n_ant_spans)
                same_speakers = same_speakers.to(torch.long)
                same_speaker_embs = self.embed_same_speaker(same_speakers) # (n_top_spans, n_ant_spans, feat_dim)

                # Embed the document genre
                genre_embs = self.embed_genre(genre) # (feat_dim,)
                genre_embs = torch.unsqueeze(torch.unsqueeze(genre_embs, 0), 0).repeat(n_top_spans, n_ant_spans, 1) # (n_top_spans, n_ant_spans, feat_dim)

                if self.span_distance_type == "segment":
                    # Embed segment-level distance between spans
                    n_segs, max_seg_len = input_ids.shape[0], input_ids.shape[1]
                    token_seg_indices = torch.arange(0, n_segs, device=self.device).unsqueeze(1).repeat(1, max_seg_len) # (n_segs, max_seg_len); segment index of (i,j)-th token
                    token_seg_indices = token_seg_indices[input_mask] # (n_tokens,); segment index of i-th token
                    top_span_seg_indices = token_seg_indices[top_span_start_token_indices] # (n_top_spans,); segment index of i-th span
                    top_antecedent_seg_indices = token_seg_indices[top_span_start_token_indices[top_antecedent_indices]] # (n_top_spans, n_ant_spans); segment index of j-th antecedent of i-th span
                    top_antecedent_seg_distances = torch.unsqueeze(top_span_seg_indices, 1) - top_antecedent_seg_indices # (n_top_spans, n_ant_spans); segment-level distance between i-th span and j-th span
                    top_antecedent_seg_distances = torch.clamp(top_antecedent_seg_distances, 0, self.config['max_training_segments'] - 1) # (n_top_spans, n_ant_spans)
                    seg_distance_embs = self.embed_segment_distance(top_antecedent_seg_distances) # (n_top_spans, n_ant_spans, feat_dim)
                elif self.span_distance_type == "sentence":
                    # Embed sentence-level distance between spans
                    top_span_sent_indices = sentence_indices[top_span_start_token_indices] # (n_top_spans,); sentence id of i-th span
                    top_antecedent_sent_indices = sentence_indices[top_span_start_token_indices[top_antecedent_indices]] # (n_top_spans, n_ant_spans); sentence index of j-th antecedent of i-th span
                    top_antecedent_seg_distances = torch.unsqueeze(top_span_sent_indices, 1) - top_antecedent_sent_indices # (n_top_spans, n_ant_spans); sentence-level distance between i-th span and j-th span
                    top_antecedent_seg_distances = torch.clamp(top_antecedent_seg_distances, 0, self.config['max_training_segments'] -1 ) # (n_top_spans, n_ant_spans)
                    seg_distance_embs = self.embed_segment_distance(top_antecedent_seg_distances) # (n_top_spans, n_atn_spans, feat_dim)
                else:
                    raise Exception("Invalid span_distance_type: %s" % self.span_distance_type)

                # Embed index-level distance between spans
                # Is index-level distance meaningful???
                top_antecedent_distances = util.bucket_distance(top_antecedent_offsets) # (n_top_spans, n_ant_spans); span-index-level distance
                top_antecedent_distance_embs = self.embed_top_antecedent_distance(top_antecedent_distances) # (n_top_spans, n_ant_spans, feat_dim)

                # Concatenate
                feature_list = [same_speaker_embs, genre_embs, seg_distance_embs, top_antecedent_distance_embs]
                feature_embs = torch.cat(feature_list, dim=2) # (n_top_spans, n_ant_spans, 4 * feat_dim)
                feature_embs = self.dropout(feature_embs) # (n_top_spans, n_ant_spans, 4 * feat_dim)

            # top-ranked spans (lhs) x top-ranked antecedents (rhs) x their product
            lhs_embs = torch.unsqueeze(top_span_embs, 1).repeat(1, n_ant_spans, 1) # (n_top_spans, n_ant_spans, span_dim)
            rhs_embs = top_span_embs[top_antecedent_indices]  # (n_top_spans, n_ant_spans, span_dim)
            product_embs = lhs_embs * rhs_embs # (n_top_spans, n_ant_spans, span_dim)

            # Concat
            if self.use_features:
                pair_embs = torch.cat([lhs_embs, rhs_embs, product_embs, feature_embs], 2) # (n_top_spans, n_ant_spans, 3*span_dim + 4*feat_dim)
            else:
                pair_embs = torch.cat([lhs_embs, rhs_embs, product_embs], 2) # (n_top_spans, n_ant_spans, 3*span_dim)

            ###################################
            # Get fine-grained coreference scores
            ###################################

            top_antecedent_scores_fine = torch.squeeze(self.mlp_coref_score(pair_embs), 2) # (n_top_spans, n_ant_spans)
            top_antecedent_scores = top_antecedent_scores_coarse + top_antecedent_scores_fine # (n_top_spans, n_ant_spans)

        else:
            top_antecedent_scores = top_antecedent_scores_coarse # (n_top_spans, n_ant_spans)

        # Append a constrant 0 vector to the first column
        top_antecedent_scores = torch.cat([torch.zeros(n_top_spans, 1, device=self.device), top_antecedent_scores], dim=1) # (n_top_spans, 1+n_ant_spans)

        if not get_loss:
            return [top_span_start_token_indices, \
                    top_span_end_token_indices, \
                    top_antecedent_indices, \
                    top_antecedent_scores], None

        ###################################
        # <9> Compute losses
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

    def _extract_top_spans(self,
                           cand_span_indices_sorted,
                           cand_span_start_token_indices,
                           cand_span_end_token_indices,
                           n_top_spans):
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

