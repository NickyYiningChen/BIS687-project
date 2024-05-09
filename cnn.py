#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import sys
from gen_embed import generate_embeddings



class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.relu = nn.ReLU ( )
        self.lstm_size = args.lstm_size 
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.embed_size = args.embed_size
        self.output_size = 1
        self.bn = nn.BatchNorm1d(self.embed_size)
        layers = [1, 2, 2]
        block = Residual
        self.layer = self.layer_generate(block, self.embed_size, layers[0], 2)


        self.vocab_embedding = nn.Embedding (args.unstructure_size+10, self.embed_size )
        self.vocab_layer = nn.Sequential(
                nn.Dropout(0.2),
                convolutional(self.embed_size, self.embed_size, 2, 2),
                nn.BatchNorm1d(self.embed_size),
                nn.Dropout(0.2),
                nn.ReLU(),
                convolutional(self.embed_size, self.embed_size, 2, 3),
                nn.BatchNorm1d(self.embed_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                )
        self.vocab_output = nn.Sequential (
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear (self.embed_size * 2, self.embed_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear ( self.embed_size, self.output_size),
        )

        self.value_embedding = nn.Embedding.from_pretrained(generate_embeddings(self.embed_size, args.split_num + 1))
        self.value_mapping = nn.Sequential(
                nn.Linear ( self.embed_size * 2, self.embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                )
        
        self.demo_embedding = nn.Embedding (args.n_ehr, self.embed_size )
        self.demo_mapping = nn.Sequential(
                nn.Linear ( self.embed_size, self.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                nn.Linear ( self.embed_size, self.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                )

        self.combiner = nn.Sequential (
            nn.Linear (args.lstm_size * 3, args.lstm_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear (args.lstm_size, self.output_size),
        )

    def embedd_value(self, x):
        indices, values = x
        embedded_indices = self.vocab_embedding(indices.view(-1))  # Flatten the indices before embedding
        embedded_values = self.value_embedding(values.view(-1))    # Flatten the values before embedding
        concatenated_embeddings = torch.cat((embedded_indices, embedded_values), dim=1)
        mapped_embeddings = self.value_mapping(concatenated_embeddings)
        final_embeddings = mapped_embeddings.view(*indices.size(), -1)  # Preserve the original batch and sequence dimensions

        return final_embeddings


    def layer_generate(self, block, out_channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.embed_size != out_channels:
            downsample = nn.Sequential(
                convolutional(self.embed_size, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        layers.append(block(self.embed_size, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def process_value_embedding(self, v):
        v = self.embedd_value(v)  # Assuming embedd_value is a defined method or layer
        v = self.visit_pooling(v)  # Assuming visit_pooling is defined
        v = torch.transpose(v, 1, 2).contiguous()
        v = self.bn(v)
        v = self.relu(v)
        v = self.layer(v)
        v = self.pooling(v)
        v = v.view((v.size(0), -1))
        return v

    def process_demographic_data(self, demo):
        if len(demo.size()) > 1:
            dsize = list(demo.size()) + [-1]
            d = self.demo_embedding(demo.view(-1)).view(dsize)
            d = self.demo_mapping(d)
            d = torch.transpose(d, 1, 2).contiguous()
            d = self.pooling(d)
            d = d.view((d.size(0), -1))
        else:
            d = torch.zeros_like(d)
        return d

    def process_notes(self, notes):
        if notes is not None:
            notes = self.vocab_layer(notes.transpose(1, 2))
            notes = self.pooling(notes)
            notes = notes.view((notes.size(0), -1))
        else:
            notes = torch.zeros((notes.size(0), self.output_size), device=notes.device)
        return notes

    def forward(self, x, t, dd, notes=None):

        x = self.process_value_embedding(x)
        d = self.process_demographic_data(dd)
        if notes is not None:
            notes = self.process_notes(notes)
        combined = torch.cat((x, notes, d), 1)
        out = self.combiner(combined)  # Assuming combiner is a defined method or layer

        return out
    
    def visit_pooling(self, tensor):

        tensor_size = tensor.size()
        merged_tensor = tensor.view(tensor_size[0] * tensor_size[1], tensor_size[2], tensor.size(3))
        transposed_tensor = merged_tensor.transpose(1, 2).contiguous()
        pooled_tensor = self.pooling(transposed_tensor)
        reshaped_tensor = pooled_tensor.view(tensor_size[0], tensor_size[1], tensor_size[3])

        return reshaped_tensor

def convolutional(in_channels, out_channels, stride=1, kernel_size=3):
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False)
    return conv

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.convolution_layers = nn.Sequential(
            convolutional(in_channels, out_channels, stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            convolutional(out_channels, out_channels)
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.convolution_layers(x)
        x += residual
        return self.final_relu(x)