import torch
from torch.utils.data import Dataset
import cv2
import os
import timm
import torch.nn as nn


class RCNN(nn.Module):
    def __init__(self, num_classes, architecture_name, pretrained, hidden_size, num_layers):
        super().__init__()
        self.feature_extractor = timm.create_model(architecture_name, pretrained=pretrained, num_classes=0)
        num_features = self.feature_extractor.num_features
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.feature_extractor(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        lin_out = self.linear(r_out[:, -1, :])
        return lin_out



my_net = RCNN(20, 'resnet18', True, 64, 2)
a = my_net(torch.randn(5, 2, 3, 224, 224))
print(a.shape)
# feature_extractor, num_features = get_model_features('resnet18', True)