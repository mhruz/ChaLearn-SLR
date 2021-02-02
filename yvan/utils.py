import torch
from torch.utils.data import Dataset
import cv2
import os
import timm
import torch.nn as nn


class DatasetFromImages(Dataset):
    # Loader from train_list.csv
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['img_list'].values
        self.labels = df['label'].values
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_images(self, img_list, transform=None):
        X = []
        img_list = img_list.split(',')
        for file_name in img_list:
            file_path = os.path.join(self.data_path, file_name)
            image = cv2.imread(file_path)

            if transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, idx):
        img_list = self.file_names[idx]
        X = self.read_images(img_list, self.transform)
        y = torch.tensor(self.labels[idx]).long()
        return X, y


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
