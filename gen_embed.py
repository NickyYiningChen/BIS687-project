#!/usr/bin/env python
# coding=utf-8
import numpy as np

import os
import torch


def generate_embeddings(dimension=200, split=200):

    positional_values = np.array([np.arange(split) * i for i in range(int(dimension / 2))], dtype=np.float32).transpose()
    positional_values /= positional_values.max()
    
    sin_embeddings = np.sin(positional_values)
    cos_embeddings = np.cos(positional_values)
    embeddings = np.concatenate((sin_embeddings, cos_embeddings), axis=1)
    embeddings[0, :dimension] = 0
    embeddings = torch.from_numpy(embeddings)
    
    return embeddings