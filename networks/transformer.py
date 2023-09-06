'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 15:33:21
Contact: yangbitao001@ke.com
'''


# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, d_k, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)     
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, mlp_ratio = 4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(d_model, d_model * mlp_ratio, dropout = dropout)
    
        self.norm1 = SLN(d_model)
        self.norm2 = SLN(d_model)
        self.norm3 = SLN(d_model) 
        self.norm4 = SLN(d_model) 
            

    def forward(self, q, k, v, z, mask=None):

        q = self.norm1(q, z)
        #k = self.norm3(k, z)
        #v = self.norm4(v, z)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)     
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) 
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual
        output = self.mlp(self.norm2(output, z)) + output
        
        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.w_2(F.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, query, key, value, z, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            query, key, value, z, mask=slf_attn_mask)
        #enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerUnit(nn.Module):
    def __init__(
            self, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0):
        super(TransformerUnit, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm = SLN(d_model) 

    def forward(self, query, key, value, z, src_mask=None, return_attns=False):

        for enc_layer in self.layer_stack:
            query, enc_slf_attn = enc_layer(query, key, value, z, slf_attn_mask=src_mask)
        query = self.norm(query, z)
        return query
