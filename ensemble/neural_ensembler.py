import torch.nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Linear, Parameter
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import os

import wandb


class AUTSLDataSet(Dataset):
    def __init__(self, csv_filenames, csv_labels, device):
        super(AUTSLDataSet, self).__init__()

        with open(csv_labels) as txt_file:
            data_buffer = txt_file.readlines()

        self.labels = []

        for i, label in enumerate(data_buffer):
            self.labels.append(int(label.split(',')[-1][:-1]))

        self.model_predicts = {}
        self.num_classes = -1
        self.num_datapoints = -1

        for pcsv in csv_filenames:
            _csv = pd.read_csv(pcsv)
            self.model_predicts[pcsv] = _csv.to_numpy()
            if self.num_classes == -1:
                self.num_classes = _csv.shape[1]
            else:
                if _csv.shape[1] != self.num_classes:
                    raise "Inconsistent number of classes in models. Fault at {}.".format(pcsv)

            if self.num_datapoints == -1:
                self.num_datapoints = _csv.shape[0]
            else:
                if _csv.shape[0] != self.num_datapoints:
                    raise "Inconsistent number of datapoints in models. Fault at {}.".format(pcsv)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.zeros((len(self.model_predicts), self.num_classes))
        for i, model_predict in enumerate(self.model_predicts):
            data[i, :] = torch.Tensor(self.model_predicts[model_predict][idx])

        label = torch.squeeze(torch.LongTensor(data=[self.labels[idx]]))

        return data.to(device), label.to(device)


class NeuralEnsemblerBERT(torch.nn.Module):
    def __init__(self, num_models, num_classes, num_heads, num_per_head, num_enc_layers=3):
        super(NeuralEnsemblerBERT, self).__init__()

        self.embed_dim = num_per_head * num_heads

        encoder_layer = TransformerEncoderLayer(self.embed_dim, num_heads, 128, activation="relu")
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers)
        self.class_token = Parameter(torch.rand(self.embed_dim))
        self.pos_embedding = Parameter(torch.rand((num_models + 1, self.embed_dim)))
        self.embedding = Linear(num_classes, self.embed_dim, bias=False)
        # self.class_head = torch.nn.Sequential(
        #     Linear(self.embed_dim, self.embed_dim),
        #     Linear(self.embed_dim, num_classes)
        # )
        self.class_head = Linear(self.embed_dim, num_classes)

        # torch.nn.init.uniform_(self.class_token)
        # torch.nn.init.xavier_normal_(self.pos_embedding)
        # torch.nn.init.normal_(self.embedding.weight)
        # torch.nn.init.normal_(self.class_head.weight)

    def forward(self, x):

        n, seq, dim = x.shape
        x = self.embedding(x)
        cls_emb = torch.tile(self.class_token, [n, 1, 1])
        pos_emb = torch.tile(self.pos_embedding, [n, 1, 1])
        t = torch.cat((cls_emb, x), 1) + pos_emb
        # reshape for encoder
        t = t.permute(1, 0, 2)
        y = self.encoder(t)

        cls_logits = self.class_head(y[0, :, :])
        # cls_softmax = F.softmax(cls_logits, dim=1)

        return cls_logits


def train(model, data_loader, epochs, optimizer, criterion):
    wandb.watch(model)

    for epoch in range(epochs):
        model.train()
        for data, label in data_loader:
            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss})

        validate(model, data_loader, criterion)


def validate(model, data_loader, criterion):
    model.eval()
    hits = 0
    num_batches = 0
    loss_val = 0

    for data, label in data_loader:
        pred = model(data)
        loss_val += criterion(pred, label)
        pred_idx = torch.argmax(pred, dim=1)
        hits += torch.sum(pred_idx == label)

        num_batches += 1

    loss_val /= num_batches
    acc = hits / len(data_loader.dataset)

    wandb.log({"loss_val": loss_val})
    wandb.log({"acc": acc})


if __name__ == "__main__":
    learning_rate = 1e-3
    epochs = 200
    batch_size = 64
    num_heads = 3
    num_per_head = 16

    # wandb.login(key="3a6eb272feb7d39068e66471c971b3ce5e45e4ab")
    wandb.init(project="my-test-project", entity="mhruz")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_dir = r"e:\ZCU\JSALT2020\ensemble_SL_sensors_2022"
    predicted_csv = os.listdir(data_dir)
    predicted_csv = [os.path.join(data_dir, pred) for pred in predicted_csv if pred.endswith(".csv")]
    val_data = AUTSLDataSet(predicted_csv, r"e:\ZCU\JSALT2020\ensemble_SL_sensors_2022\AUTSL_val.txt", device)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    ensembler = NeuralEnsemblerBERT(14, val_data.num_classes, num_heads, num_per_head)
    ensembler = ensembler.to(device)

    # optimizer = SGD(ensembler.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = Adam(ensembler.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    train(ensembler, val_data_loader, epochs, optimizer, criterion)
