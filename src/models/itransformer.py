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

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


    

class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, time_dim, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.time_embed = nn.Linear(time_dim, embed_dim)

    def forward(self, x, t):
        time_embed = self.time_embed(t).unsqueeze(1)  # Shape: [batch, 1, embed_dim]
        return self.norm(x + time_embed)
    

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, time_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x dimension ]
            t : [ batch_size x time_dim ]
            returns:
            out : [ batch_size x out_channels x dimension ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t) 
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


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
    def __init__(self, attn_layers, time_dim, hidden_size, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.time_res = TimeConditionedResidualBlock(time_dim, hidden_size)
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, t, attn_mask=None):
        # x [B, N, E]
        x = self.time_res(x, t)
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, t, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, t, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        hidden_size,
        time_dim,
        conv_hidden_size=None,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = ResidualTemporalBlock(
            inp_channels=hidden_size, out_channels=conv_hidden_size, time_dim=time_dim, kernel_size=1
        )
        self.conv2 = ResidualTemporalBlock(
            inp_channels=conv_hidden_size, out_channels=hidden_size, time_dim=time_dim, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, t, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1),t)))
        y = self.dropout(self.conv2(y,t).transpose(-1, 1))

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

    def __init__(self, c_in, hidden_size, time_dim, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding1 = nn.Linear(c_in, hidden_size)
        self.value_embedding2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, hidden_size),
            Rearrange('batch t -> batch 1 t'),
        )

    def forward(self, x, x_mark,t=None):

        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding1(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding1(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate hidden_size]
        if t is not None:
            a = self.time_mlp(t) 
            x = x + self.time_mlp(t) 
            x = self.dropout(F.gelu(x))
            x = self.value_embedding2(x)
        else:
            x = self.dropout(x)
        return x
        

class DataEmbedding_inverted_context(nn.Module):
    """
    DataEmbedding_inverted_context
    """

    def __init__(self, c_in, hidden_size, dropout=0.1):
        super(DataEmbedding_inverted_context, self).__init__()
        self.value_embedding1 = nn.Linear(c_in, hidden_size)
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
    


class iTransformerModel(nn.Module):

    def __init__(self, 
        horizon,
        context_len,
        data_dim,
        # len_lags_seq,
        time_dim: int=32,
        hidden_size: int = 1024,
        n_heads: int = 8,
        e_layers: int = 4,
        d_ff: int = 4096,
        factor: int = 1,
        dropout: float = 0.1,
        **trainer_kwargs
        ):
        super(iTransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.data_dim = data_dim
        # self.len_lags_seq = len_lags_seq
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            horizon, self.hidden_size//2, 32, self.dropout
        )
        self.enc_embedding_context = DataEmbedding_inverted_context(context_len, self.hidden_size//2, self.dropout)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
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
                    time_dim,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=F.gelu,
                )
                for l in range(self.e_layers)
            ],
            time_dim = time_dim,
            hidden_size = hidden_size,
            norm_layer=torch.nn.LayerNorm(self.hidden_size),
        )
        self.projector = nn.Linear(self.hidden_size, horizon, bias=True)

        
    def forward(self, xt, t, xc, timefeats_c, timefeats_p):
        '''
            xt : [ batch , horizon , N ]
        '''
        t = self.time_mlp(t)
        x=xt

        _, _, N = x.shape  # B L N
        # B: batch_size;       E: hidden_size;
        # L: input_size;       S: horizon(h);
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out_x = self.enc_embedding(
            x, timefeats_p, t
        )  # covariates (e.g timestamp) can be also embedded as tokens
        enc_out_context = self.enc_embedding_context(xc, timefeats_c)
        enc_out = torch.cat([enc_out_context, enc_out_x],-1)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, t, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates


        # domain_map might have squeezed the last dimension in case input_dim == 1
        if dec_out.ndim == 2:
            return dec_out.unsqueeze(-1)
        else:
            return dec_out    


        



