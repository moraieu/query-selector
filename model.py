import time
from multiprocessing import Pool

import numpy
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1))
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)


def attention(Q, K, V):
    a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)
    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class QuerySelector(nn.Module):
    def __init__(self, fraction=0.33):
        super(QuerySelector, self).__init__()
        self.fraction = fraction

    def forward(self, queries, keys, values):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape
        l_Q = int((1.0 - self.fraction) * L_Q)
        K_reduce = torch.mean(keys.topk(l_Q, dim=1).values, dim=1).unsqueeze(1)
        sqk = torch.matmul(K_reduce, queries.transpose(1,2))
        indices = sqk.topk(l_Q, dim=-1).indices.squeeze(1)
        Q_sample = queries[torch.arange(B)[:, None], indices, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_sample, keys.transpose(-2, -1))
        attn = torch.softmax(Q_K / math.sqrt(D), dim=-1)
        mean_values = values.mean(dim=-2)
        result = mean_values.unsqueeze(-2).expand(B, L_Q, mean_values.shape[-1]).clone()
        result[torch.arange(B)[:, None], indices, :] = torch.matmul(attn, values).type_as(result)
        return result, None

    def inference(self):
        pass # no parameters


class InferenceModule(torch.nn.Module):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()



class InferenceModuleList(torch.nn.ModuleList):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()


class AttentionBlock(InferenceModule):
    def __init__(self, dim_val, dim_attn, debug=False, attn_type='full'):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
        self.debug = debug
        self.qk_record = None
        self.qkv_record = None
        self.n = 0
        if attn_type == "full":
            self.attentionLayer = None
        elif attn_type.startswith("query_selector"):
            args = {}
            if len(attn_type.split('_')) == 3:
                args['fraction'] = float(attn_type.split('_')[-1])
            self.attentionLayer = QuerySelector(**args)
        else:
            raise Exception

    def forward(self, x, kv=None):
        if kv is None:
            if self.attentionLayer:
                qkv = self.attentionLayer(self.query(x), self.key(x), self.value(x))[0]
            else:
                qkv = attention(self.query(x), self.key(x), self.value(x))
            return qkv
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(InferenceModule):
    def __init__(self, dim_val, dim_attn, n_heads, attn_type):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn, attn_type=attn_type))

        self.heads = InferenceModuleList(self.heads)
        self.fc = Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))

        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs

        x = self.fc(a)

        return x

    def record(self):
        for h in self.heads:
            h.record()


class Value(InferenceModule):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        return self.fc1(x)


class Key(InferenceModule):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        return self.fc1(x)


class Query(InferenceModule):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        return self.fc1(x)


class PositionalEncoding(InferenceModule):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class EncoderLayer(InferenceModule):
    def __init__(self, dim_val, dim_attn, n_heads=1, attn_type='full'):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads, attn_type=attn_type)

        self.fc1 = Linear(dim_val, dim_val)
        self.fc2 = Linear(dim_val, dim_val)

        self.norm1 = LayerNorm(dim_val)
        self.norm2 = LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)

        return x

    def record(self):
        self.attn.record()


class DecoderLayer(InferenceModule):
    def __init__(self, dim_val, dim_attn, n_heads=1, attn_type='full'):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads, attn_type=attn_type)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads, attn_type=attn_type)

        self.fc1 = Linear(dim_val, dim_val)
        self.fc2 = Linear(dim_val, dim_val)

        self.norm1 = LayerNorm(dim_val)
        self.norm2 = LayerNorm(dim_val)
        self.norm3 = LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a)
        return x

    def record(self):
        self.attn1.record()
        self.attn2.record()


class Dropout(nn.Dropout):
    def forward(self, x=False):
        if self.training:
            return super(Dropout, self).forward(x)
        else:
            return x

    def inference(self):
        self.training = False


class Linear(nn.Linear):
    def forward(self, x=False):
        if self.training:
            return super(Linear, self).forward(x)
        else:
            return F.linear(x, self.weight.data, self.bias.data if self.bias is not None else None)

    def inference(self):
        self.training = False


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if self.training:
            return super(LayerNorm, self).forward(x)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight.data, self.bias.data, self.eps)

    def inference(self):
        self.training = False


class Transformer(InferenceModule):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1,
                 enc_attn_type='full', dec_attn_type='full', n_heads=1, dropout=0.1, debug=False, output_len=1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        self.output_len = output_len

        # Initiate encoder and Decoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads, attn_type=enc_attn_type))
        self.encs = InferenceModuleList(self.encs)
        self.decs = []
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads, attn_type=dec_attn_type))
        self.decs = InferenceModuleList(self.decs)
        self.pos = PositionalEncoding(dim_val)

        self.enc_dropout = Dropout(dropout)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = Linear(input_size, dim_val)
        self.dec_input_fc = Linear(input_size, dim_val)
        self.out_fc = Linear(dec_seq_len * dim_val, out_seq_len*output_len)

        self.debug = debug

    def forward(self, x):
        # encoder
        e = self.encs[0](self.pos(self.enc_dropout(self.enc_input_fc(x))))

        for enc in self.encs[1:]:
            e = enc(e)
        if self.debug:
            print('Encoder output size: {}'.format(e.shape))
        # decoder
        decoded = self.dec_input_fc(x[:, -self.dec_seq_len:])

        d = self.decs[0](decoded, e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        x = self.out_fc(d.flatten(start_dim=1))
        return torch.reshape(x, (x.shape[0], -1, self.output_len))

    def record(self):
        self.debug = True
        for enc in self.encs:
            enc.record()
        for dec in self.decs:
            dec.record()


