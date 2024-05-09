import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np

class Loss(nn.Module):
    def __init__(self, enable_mining):
        super(Loss, self).__init__()
        self.loss_fn = nn.BCELoss()
        self.enable_mining = enable_mining
        self.activation_fn = nn.Sigmoid()
    
    def selective_mining(score_output, score_labels, threshold_count, is_largest=True):
        threshold_count = min(max(threshold_count, 10), len(score_output))
        _, indices = torch.topk(score_output, min(threshold_count, len(score_output)), largest=is_largest)
        score_output = torch.index_select(score_output, 0, indices)
        score_labels = torch.index_select(score_labels, 0, indices)
        return score_output, score_labels

    def forward(self, predictions, targets, training=True):
        predictions = self.activation_fn(predictions)

        positive_indices = targets > 0.5
        negative_indices = targets < 0.5
        positive_labels = targets[positive_indices]
        negative_labels = targets[negative_indices]
        positive_predictions = predictions[positive_indices]
        negative_predictions = predictions[negative_indices]
        positive_loss, negative_loss = 0, 0

        # selective mining
        pos_threshold = 10
        neg_threshold = 18
        if self.enable_mining:
            positive_predictions, positive_labels = selective_mining(positive_predictions, positive_labels, pos_threshold, is_largest=False)
            negative_predictions, negative_labels = selective_mining(negative_predictions, negative_labels, neg_threshold, is_largest=True)

        if len(positive_predictions):
            positive_loss = 0.5 * self.loss_fn(positive_predictions, positive_labels) 

        if len(negative_predictions):
            negative_loss = 0.5 * self.loss_fn(negative_predictions, negative_labels)
        total_loss = positive_loss + negative_loss
        
        predictions = predictions.data.cpu().numpy() > 0.5
        targets = targets.data.cpu().numpy()
        pos_true = (targets == 1).sum()
        neg_true = (targets == 0).sum()
        pos_correct = (predictions + targets == 2).sum()
        neg_correct = (predictions + targets == 0).sum()

        return [total_loss, pos_correct, pos_true, neg_correct, neg_true]
