import torch
import torch.nn as nn
import einops
import math
import inspect
from torch import Tensor
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from math import sqrt
import numpy as np
# from functools import partial, wraps
# from dataclasses import dataclass, fields, is_dataclass
from typing import Optional, List, Tuple, Set, Callable, Union, Any
# from collections import OrderedDict
from .helpers import (
    TimestepEmbedding,
    SinusoidalPosEmb,
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
    Residual,
    PreNorm,
    LinearAttention,
)



class TriangularCausalMask:
    """
    TriangularCausalMask
    """

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class Encoder(nn.Module):
    def __init__(self, attn_layers, hidden_size, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, N, E]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        hidden_size,
        conv_hidden_size=None,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=conv_hidden_size, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size, out_channels=hidden_size, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn



class AttentionLayer(nn.Module):
    def __init__(self, attention, hidden_size, n_head, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_head)
        d_values = d_values or (hidden_size // n_head)

        self.inner_attention = attention
        self.query_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.value_projection = nn.Linear(hidden_size, d_values * n_head)
        self.out_projection = nn.Linear(d_values * n_head, hidden_size)
        self.n_head = n_head

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class FullAttention(nn.Module):
    """
    FullAttention
    """

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

        

class DataEmbedding_inverted(nn.Module):
    """
    DataEmbedding_inverted
    """

    def __init__(self, c_in, hidden_size, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding1 = nn.Linear(c_in, hidden_size)
        self.value_embedding2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        
        if x_mark is None:
            x = self.value_embedding1(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding1(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate hidden_size]
        x = self.dropout(x)
        return x
    
# class DataEmbedding_inverted(nn.Module):
#     """
#     DataEmbedding_inverted
#     """

#     def __init__(self, c_in1, c_in2, hidden_size, dropout=0.1):
#         super(DataEmbedding_inverted, self).__init__()
#         self.value_embedding1 = nn.Linear(c_in1, hidden_size)
#         self.value_embedding3 = nn.Linear(c_in2, hidden_size)
#         self.value_embedding2 = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x, x_mark):

#         x = x.permute(0, 2, 1) # x: [Batch Variate_x Time]
        
#         x = self.value_embedding1(x) # x: [Batch Variate_x hidden_size]
#         if x_mark is not None:
#             # the potential to take covariates (e.g. timestamps) as tokens
#             x_mark = self.value_embedding3(x_mark.permute(0, 2, 1)) # x_mark: [Batch Variate_xm hidden_size]
#             x = self.value_embedding2(torch.cat([x, x_mark], 1))
#         # x: [Batch Variate hidden_size]
#         x = self.dropout(x)
#         return x

class iTransformerModelDeter(nn.Module):

    def __init__(self, 
        horizon,
        context_len,
        # len_lags_seq,
        # input_dim,
        data_dim,
        hidden_size: int = 1024,
        n_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = None,
        factor: int = 1,
        dropout: float = 0.1,
        **trainer_kwargs
        ):
        super(iTransformerModelDeter, self).__init__()
        # self.enc_in = input_dim
        # self.dec_in = input_dim
        # self.c_out = input_dim
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = num_layers
        if d_ff is None:
            self.d_ff = hidden_size*4
        else:
            self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.data_dim = data_dim
        # Embedding
        # self.enc_embedding = DataEmbedding_inverted(
        #     context_len*(len_lags_seq+1), self.hidden_size, self.dropout
        # )
        self.enc_embedding = DataEmbedding_inverted(
            context_len, self.hidden_size, self.dropout
        )

        self.horizon = horizon

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, self.factor, attention_dropout=self.dropout
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=F.gelu,
                )
                for l in range(self.e_layers)
            ],
            hidden_size = hidden_size,
            norm_layer=torch.nn.LayerNorm(self.hidden_size),
        )
        self.projector = nn.Linear(self.hidden_size, horizon, bias=True)

        
    def forward(self, x, timefeats=None, *args, **kwargs):
        '''
            x : [ batch x horizon x dimension ]
        '''

        _, _, N = x.shape  # B L N
        # B: batch_size;       E: hidden_size;
        # L: input_size;       S: horizon(h);
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(
            x, timefeats
        )  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates


        dec_out = dec_out[:, -self.horizon:, :self.data_dim]

        # domain_map might have squeezed the last dimension in case input_dim == 1
        if dec_out.ndim == 2:
            return dec_out.unsqueeze(-1)
        else:
            return dec_out    


        



