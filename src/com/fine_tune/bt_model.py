"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

 BTM model
"""
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from src.com.pre_train.pretrain_model import BrainBertPreTrainedModel, BrainBertModel


class BrainBertForBTRetrieval(BrainBertPreTrainedModel):
    """ Finetune for Brain text retrieval: BTRC and BTM
    """

    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.brainbert = BrainBertModel(config, img_dim)
        self.dropout = nn.Dropout()
        self.btm_output = nn.Linear(config.hidden_size, 2)  # FC Layer
        # self.rank_output = nn.Linear(config.hidden_size, config.vocab_size)  # non-linear layer
        self.rank_output = nn.Linear(config.vocab_size, 1)  # non-linear layer
        self.margin = margin
        self.apply(self.init_weights)

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.btm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.btm_output.bias.data[1:]

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        if task == 'btm':
            sample_size = batch['sample_size']
            return self.btm_forward(input_ids, position_ids, img_feat, img_pos_feat, attention_mask, gather_index,
                                    sample_size, compute_loss)
        if task == 'btrc':
            sample_size = batch['sample_size']
            return self.btrc_forward(input_ids, position_ids, img_feat, img_pos_feat, attention_mask, gather_index,
                                     sample_size, compute_loss)

    def btm_forward(self, input_ids, position_ids, img_feat, img_pos_feat, attention_mask, gather_index, sample_size,
                    compute_loss):
        sequence_output = self.brainbert(input_ids, position_ids,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False)
        pooled_output = self.brainbert.pooler(sequence_output)
        rank_scores = self.btm_output(pooled_output)
        if compute_loss:
            # triplet loss
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
            pos = scores[:, :1]
            neg = scores[:, 1:]
            rank_loss = torch.clamp(self.margin + neg - pos, 0)
            return rank_loss
        else:
            return rank_scores

    def btrc_forward(self, input_ids, position_ids, img_feat, img_pos_feat, attention_mask, gather_index, sample_size,
                     compute_loss):
        sequence_output = self.brainbert(input_ids, position_ids,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False)
        pooled_output = self.brainbert.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        if compute_loss:
            # triplet loss
            norm_score = self.dropout(F.softmax(rank_scores, dim=0))
            output = norm_score.T.matmul(input_ids.float()).squeeze(1)
            return output
            # rank_scores_softmax = torch.softmax(rank_scores)
            #
            # scores = rank_scores_softmax.contiguous().view(-1, sample_size)
            # pos = scores[:, :1]
            # neg = scores[:, 1:]
            # rank_loss = torch.clamp(self.margin + neg - pos, 0)
            # return rank_loss
        else:
            return rank_scores


class BrainBertForBTRetrievalHardNeg(BrainBertForBTRetrieval):
    """ Finetune UNITER for image text retrieval
    """

    def __init__(self, config, img_dim, margin=0.2, hard_size=16):
        super().__init__(config, img_dim, margin)
        self.hard_size = hard_size

    def forward(self, batch, sample_from='t', compute_loss=True):
        # expect same input_ids for all pairs
        batch_size = batch['attn_masks'].size(0)
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        if sample_from == 't':
            if input_ids.size(0) == 1:
                batch['input_ids'] = input_ids.expand(batch_size, -1)
        elif sample_from == 'i':
            if img_feat.size(0) == 1:
                batch['img_feat'] = img_feat.expand(batch_size, -1, -1)
            if img_pos_feat.size(0) == 1:
                batch['img_pos_feat'] = img_pos_feat.expand(batch_size, -1, -1)
        else:
            raise ValueError()

        if self.training and compute_loss:
            with torch.no_grad():
                self.eval()
                scores = super().forward(batch, compute_loss=False)
                hard_batch = self._get_hard_batch(batch, scores, sample_from)
                self.train()
            return super().forward(hard_batch, compute_loss=True)
        else:
            return super().forward(batch, compute_loss)

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(
            self.hard_size, sorted=False)[1] + 1
        indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                         device=hard_indices.device),
                             hard_indices])

        attention_mask = attention_mask.index_select(0, indices)
        gather_index = gather_index.index_select(0, indices)
        if position_ids.size(0) != 1:
            position_ids = position_ids[:self.hard_size + 1]

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(dim=1).max().item()
            max_i = max_len - input_ids.size(1)
            attention_mask = attention_mask[:, :max_len]
            gather_index = gather_index[:, :max_len]
            img_feat = img_feat.index_select(0, indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(0, indices)[:, :max_i, :]
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size + 1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(0, indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size + 1]
            img_pos_feat = img_pos_feat[:self.hard_size + 1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['gather_index'] = gather_index

        return hard_batch
