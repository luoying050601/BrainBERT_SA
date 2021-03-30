"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import torch
import copy
import json
import logging
from torch import nn
from io import open
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from torch.nn import functional as F
from collections import defaultdict
from apex.normalization.fused_layer_norm import FusedLayerNorm
from src.com.pre_train.layer import BertLayer, BertPooler, GELU, BertOnlyMLMHead

logger = logging.getLogger(__name__)


class BrainBertConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=1024,
                 num_hidden_layers=24,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = BrainBertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BrainBertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BrainBertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        #  from transformers import AutoModel
        #         model = AutoModel.from_pretrained("bert-large-uncased")
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Brain class
        """
        # Load config
        config = BrainBertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        # state_dict  =
        # model.load_state_dict(
        #     {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()})

        for key in state_dict.keys():
            key = key.replace('module.', '')
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                model.__class__.__name__,
                "\n\t".join(error_msgs)))
        return model

#
# class RegionFeatureRegression(nn.Module):
#     """ for MRM"""
#
#     def __init__(self, hidden_size, feat_dim, img_linear_weight):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
#                                  GELU(),
#                                  LayerNorm(hidden_size, eps=1e-12))
#
#         self.weight = img_linear_weight
#         self.bias = nn.Parameter(torch.zeros(feat_dim))
#
#     def forward(self, input_):
#         hidden = self.net(input_)
#         output = F.linear(hidden, self.weight.t(), self.bias)
#         return output
#
#
# class RegionClassification(nn.Module):
#     """ for MRC(-kl)"""
#
#     def __init__(self, hidden_size, label_dim):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
#                                  GELU(),
#                                  LayerNorm(hidden_size, eps=1e-12),
#                                  nn.Linear(hidden_size, label_dim))
#
#     def forward(self, input_):
#         output = self.net(input_)
#         return output
#

def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.uint8, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask.bool()).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill(x_pad.bool(), 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill(joint_pad.bool(), 0)
    A.masked_fill(joint_pad.bool(), 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill(joint_pad.bool(), 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=1024, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill(joint_pad.bool(), 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance


def _compute_masked_hidden(hidden, mask):
    """ get only the masked region (don't compute unnecessary hiddens) """
    mask = mask.unsqueeze(-1).expand_as(hidden)
    # print(hidden[mask].size(0)/hidden.size(-1))
    hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))# reshape (-1,1024)
    return hidden_masked


class BrainBertForPreTraining(BrainBertPreTrainedModel):
    """ BrainBert pretraining """

    def __init__(self, config, img_dim):
        super().__init__(config)
        self.brainbert = BrainBertModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.brainbert.embeddings.word_embeddings.weight)
        # self.feat_regress = RegionFeatureRegression(
        #     config.hidden_size, img_dim,
        #     self.brainbert.img_embeddings.img_linear.weight)
        # self.region_classifier = RegionClassification(
        #     config.hidden_size, img_label_dim)
        self.btm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        if batch is not None:
            batch = defaultdict(lambda: None, batch)
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            img_feat = batch['img_feat']
            img_pos_feat = batch['img_pos_feat']
            attention_mask = batch['attn_masks']
            gather_index = batch['gather_index']
            if task == 'mlm':
                txt_labels = batch['txt_labels']
                return self.forward_mlm(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attention_mask, gather_index,
                                        txt_labels, compute_loss)
            # elif task == 'btm':
            #     targets = batch['targets']
            #     ot_inputs = batch['ot_inputs']
            #     return self.forward_btm(input_ids, position_ids,
            #                             img_feat, img_pos_feat,
            #                             attention_mask, gather_index,
            #                             targets, ot_inputs, compute_loss)

    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index,
                    txt_labels, compute_loss=True):
        sequence_output = self.brainbert(input_ids, position_ids,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        # 取出被mask的部分的概率向量
        masked_output = _compute_masked_hidden(sequence_output,
                                               txt_labels != -1)
        prediction_scores = self.cls(masked_output)
        if compute_loss:
            labels = txt_labels[txt_labels != -1]
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             labels,
                                             reduction='none')
            print(masked_lm_loss.mean())
            return masked_lm_loss
        else:
            return prediction_scores
    #
    # def forward_btm(self, input_ids, position_ids, img_feat, img_pos_feat,
    #                 attention_mask, gather_index, targets, ot_inputs,
    #                 compute_loss=True):
    #     sequence_output = self.brainbert(input_ids, position_ids,
    #                                      img_feat, img_pos_feat,
    #                                      attention_mask, gather_index,
    #                                      output_all_encoded_layers=False)
    #     pooled_output = self.brainbert.pooler(sequence_output)
    #     btm_scores = torch.sigmoid(self.btm_output(pooled_output))
    #     # OT loss
    #     if ot_inputs is not None:
    #         ot_scatter = ot_inputs['ot_scatter']
    #
    #         b = sequence_output.size(0)
    #         tl = input_ids.size(1)
    #         il = img_feat.size(1)
    #         max_l = max(ot_inputs['scatter_max'] + 1, tl + il)
    #
    #         ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
    #         ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
    #                               dtype=sequence_output.dtype,
    #                               device=sequence_output.device
    #                               ).scatter_(dim=1, index=ot_scatter,
    #                                          src=sequence_output)
    #         txt_emb = ctx_emb[:, :tl, :]
    #         img_emb = ctx_emb[:, tl:tl + il, :]
    #
    #         txt_pad = ot_inputs['txt_pad']
    #         img_pad = ot_inputs['img_pad']
    #         # NOTE: run in fp32 for stability
    #         ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
    #                                          txt_pad, img_pad).to(txt_emb)
    #         ot_pos_dist = ot_dist.masked_select((targets == 1))
    #         ot_neg_dist = ot_dist.masked_select((targets == 0))
    #         ot_loss = (ot_pos_dist, ot_neg_dist)
    #     else:
    #         ot_loss = None
    #
    #     if compute_loss:
    #         btm_loss = F.cross_entropy(btm_scores, targets, reduction='none')
    #         return btm_loss, ot_loss
    #     else:
    #         return btm_scores, ot_loss
    #

class BrainBertTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # 各种embedding
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BrainBertBrainEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        # self.pos_linear = nn.Linear(img_dim, config.hidden_size)  # pos 出错率最高，可以重点理解这里
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask
        # a =
        transformed_im = self.img_layer_norm(self.img_linear(img_feat.squeeze(-1)))
        # transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat.squeeze(-1)))
        # embeddings = transformed_im.unsqueeze(-1) + type_embeddings
        # embeddings = transformed_im.unsqueeze(-1) + transformed_pos.unsqueeze(-1) + type_embeddings
        embeddings = self.LayerNorm(transformed_im)
        embeddings = self.dropout(embeddings)
        return embeddings


class BrainBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        # attention_mask：self-attention使用。
        # 根据attention_mask做维度广播(B×H×S×S)，
        # H是head数量，此时，方便下文做self-attention时作mask，
        # 即：softmax前对logits作处理，logits+extended_attention_mask，
        # 即：attention_mask取值为1时，extended_attention_mask对应位置的取值为0；
        # 否则，attention_mask为0时，extended_attention_mask对应位置的取值为-10000.0 (很小的一个数)，
        # 这样softmax后，mask很小的值对应的位置概率接近0达到mask的目的。
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BrainBertModel(BrainBertPreTrainedModel):
    """ Modification for Joint Brain-Language Encoding
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = BrainBertTextEmbeddings(config)
        self.img_embeddings = BrainBertBrainEmbeddings(config, img_dim)
        self.encoder = BrainBertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, pos_id, img_masks=None,
                                img_type_ids=None):
        # if img_type_ids is None:
        #     # 去除第三维，保持原有1，2维形状做全1向量
        #     img_type_ids = torch.ones_like(img_feat.long())
        # img_type_embeddings = self.embeddings.token_type_embeddings(
        #     img_type_ids)
        # # TODO 这里大改
        # img_pos_embeddings = self.embeddings.position_embeddings(
        #     img_type_ids)
        # # img_feat = torch.zeros(img_feat.shape)
        img_pos_embeddings =None
        img_type_embeddings = None
        output = self.img_embeddings(img_feat, img_pos_embeddings,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)
        # TODO type id 是什么～
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, position_ids, img_masks, txt_type_ids)
        # img_emb = torch.zeros(img_feat.shape)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        # TODO 改成了hen着拼
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb.unsqueeze(0)], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):
        # compute self-attention mask
        # 第一部分，对 attention_mask 进行操作,并对输入做embedding
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            # 分别embedding 上下拼接。
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)
        # 第二部分 进入 encoder 进行编码
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        # 是否只取最后一层
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers
