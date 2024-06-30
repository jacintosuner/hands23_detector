# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from typing import Dict, Union
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

ROI_T_HEAD_REGISTRY = Registry("ROI_T_HEAD")

@ROI_T_HEAD_REGISTRY.register()
class FastRCNNFCHead(nn.Module):
    """
    A head for object tool type prediction
    """

    @configurable
    def __init__(
        self,
        input_shape: int,
    ):
        """

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)

        input_size = 1024
        self.relation_layer = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 7))
        
        self.relation_loss = nn.CrossEntropyLoss(ignore_index=100)
        

        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.relation_layer[0], 0, 0.01)
        normal_init(self.relation_layer[2], 0, 0.01)
        normal_init(self.relation_layer[4], 0, 0.01)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape
        }

    def forward(self, z_feature):
        return self.relation_layer(z_feature)

    def losses(self, relation_pred, label):
       
        losses = {
                "loss_touch": 0.1 * self.relation_loss(relation_pred, label)
        }

        return losses


def build_t_head(cfg, input_shape):
    name = "FastRCNNFCHead"
    return ROI_T_HEAD_REGISTRY.get(name)(cfg, input_shape)
