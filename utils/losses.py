import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   #目标类概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()
class BatchDynamicsLoss(nn.Module):
    def __init__(self, config):
        super(BatchDynamicsLoss, self).__init__()
        # self.cls_num_list = torch.ones(config.num_classes)
        self.cls_num_list = [math.sqrt(x)*0.5 for x in config.cls_num_list]
        self.config = config

    def forward(self, input, label,is_best = False, is_val = False):
        if is_best:
            return F.cross_entropy(input, label)

        class_counts = {}
        for i in range(self.config.num_classes):
            class_counts[i] = i/2
        weight = torch.tensor(self.cls_num_list, dtype=torch.float)
        weight = torch.sqrt(weight)
        weight = torch.flip(weight, dims=[0])
        softmax = torch.nn.Softmax(dim=0)
        weight = softmax(weight)
        weight = torch.tensor(weight.unsqueeze(0).expand_as(input),dtype=torch.float).to('cuda')

        input = input + weight * 0.5
        for i in range(self.config.num_classes):
            self.cls_num_list[i] += class_counts[i]
        return F.cross_entropy(input, label)
        # return focal_loss(F.cross_entropy(0.6*input, label, reduction='none'), 0.5)


    