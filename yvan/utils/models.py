import timm
import torch.nn as nn

class RecurrentCNN(nn.Module):
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
