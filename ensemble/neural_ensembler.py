import sys

import torch.nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Linear, Parameter, Sequential
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import os
import argparse

import wandb


def cross_entropy(y_pre, y):
    y = F.one_hot(y, num_classes=y_pre.shape[1])
    loss = -torch.sum(y * torch.log(y_pre))
    return loss / y_pre.shape[0]


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
            _csv = pd.read_csv(pcsv, sep=None)
            self.model_predicts[pcsv] = _csv.to_numpy()
            if self.num_classes == -1:
                self.num_classes = _csv.shape[1]
            else:
                if _csv.shape[1] != self.num_classes:
                    raise Exception("Inconsistent number of classes in models. Fault at {}.".format(pcsv))

            if self.num_datapoints == -1:
                self.num_datapoints = _csv.shape[0]
            else:
                if _csv.shape[0] != self.num_datapoints:
                    raise Exception("Inconsistent number of datapoints in models. Fault at {}.".format(pcsv))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.zeros((len(self.model_predicts), self.num_classes))
        for i, model_predict in enumerate(self.model_predicts):
            data[i, :] = torch.Tensor(self.model_predicts[model_predict][idx])

        label = torch.squeeze(torch.LongTensor(data=[self.labels[idx]]))

        return data.to(device), label.to(device)


class NeuralEnsemblerBERT(torch.nn.Module):
    def __init__(self, num_models, num_classes, num_heads, num_per_head, dim_feedforward=128, num_enc_layers=3):
        super(NeuralEnsemblerBERT, self).__init__()

        self.embed_dim = num_per_head * num_heads

        encoder_layer = TransformerEncoderLayer(self.embed_dim, num_heads, dim_feedforward, activation="relu")
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers)
        self.class_token = Parameter(torch.rand(self.embed_dim))
        self.pos_embedding = Parameter(torch.rand((num_models + 1, self.embed_dim)))
        self.embedding = Linear(num_classes, self.embed_dim, bias=False)
        # self.class_head = torch.nn.Sequential(
        #     Linear(self.embed_dim, self.embed_dim),
        #     Linear(self.embed_dim, num_classes)
        # )
        self.class_head = Linear(self.embed_dim, num_classes)

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


class NeuralEnsemblerBERTWeighter(torch.nn.Module):
    def __init__(self, num_models, num_classes, num_heads, num_per_head, dim_feedforward=128, num_enc_layers=3):
        super(NeuralEnsemblerBERTWeighter, self).__init__()

        self.embed_dim = num_per_head * num_heads
        self.num_models = num_models

        encoder_layer = TransformerEncoderLayer(self.embed_dim, num_heads, dim_feedforward, activation="relu")
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers)
        self.pos_embedding = Parameter(torch.rand((num_models, self.embed_dim)))
        self.embedding = Linear(num_classes, self.embed_dim, bias=False)
        self.model_weight_list = torch.nn.ModuleList()
        for i in range(num_models):
            self.model_weight_list.append(Sequential(Linear(self.embed_dim, 1), torch.nn.Sigmoid()))

    def forward(self, x):
        n, seq, dim = x.shape
        x_emb = self.embedding(x)
        pos_emb = torch.tile(self.pos_embedding, [n, 1, 1])
        t = x_emb + pos_emb
        # reshape for encoder to seq, n, dim
        t = t.permute(1, 0, 2)
        y = self.encoder(t)

        model_weights = torch.Tensor(size=(n, self.num_models))
        if y.is_cuda:
            model_weights = model_weights.to(y.get_device())

        for i in range(self.num_models):
            model_weights[:, i] = self.model_weight_list[i](y[i, :, :]).squeeze()

        model_weights /= torch.sum(model_weights, dim=1, keepdim=True)

        weighted_models = x * model_weights.unsqueeze(2).repeat([1, 1, dim])

        out = torch.sum(weighted_models, dim=1)
        out /= torch.sum(out, dim=1, keepdim=True)

        return out


def train(model, data_loader, epochs, optimizer, criterion, val_data_loader=None):
    # wandb.watch(model)

    for epoch in range(epochs):
        model.train()
        for data, label in data_loader:
            data = augment(data, apply_p=0.0, gauss_std=0.01, uncertain=0.01)

            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss})

        validate(model, val_data_loader, criterion)
        validate(model, data_loader, criterion, suffix="_train")

    return model


def validate(model, data_loader, criterion, suffix="_val"):
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

    wandb.log({"loss{}".format(suffix): loss_val})
    wandb.log({"acc{}".format(suffix): acc})


def augment(data, apply_p=0.5, gauss_std=0.01, uncertain=0.1):
    # data is expected (batch_size, seq_len, features)
    # generate apply mask
    apply_mask = torch.rand((data.shape[0], data.shape[1]))
    apply_mask = apply_mask <= apply_p
    gauss_noise = torch.normal(0, gauss_std, size=data.shape)
    if data.is_cuda:
        gauss_noise = gauss_noise.to(data.get_device())

    # compute the noisy data
    data_noise = data + gauss_noise
    # apply only to some sequence elements based on apply_p
    # TODO: better way to write this?!
    data[torch.where(apply_mask.unsqueeze(2).repeat([1, 1, data.shape[2]]) == True)] = data_noise[
        torch.where(apply_mask.unsqueeze(2).repeat([1, 1, data.shape[2]]) == True)]

    # make some models absolutely uncertain
    apply_mask = torch.rand((data.shape[0], data.shape[1]))
    apply_mask = apply_mask <= uncertain
    data[torch.where(apply_mask.unsqueeze(2).repeat([1, 1, data.shape[2]]) == True)] = 1

    data = data / torch.sum(data, dim=2).unsqueeze(2).repeat([1, 1, data.shape[2]])

    return data


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Train Neural Ensembler')
    parser.add_argument('train_dir', type=str, help='path to train dataset root')
    parser.add_argument('test_dir', type=str, help='path to test dataset root')
    parser.add_argument('train_data', type=str, help='path to train dataset')
    parser.add_argument('test_data', type=str, help='path to test dataset')
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--num_heads', type=int, help='number of heads in Encoder', default=8)
    parser.add_argument('--num_per_head', type=int, help='dimension of heads in Encoder', default=32)
    parser.add_argument('--dim_feedforward', type=int, help='dimension of heads in Encoder', default=1024)
    parser.add_argument('--num_layers', type=int, help='dimension of heads in Encoder', default=6)
    parser.add_argument('--max_epoch', type=int, help='number of max epochs', default=30)
    parser.add_argument('--batch_size', type=int, help='number data in one batch', default=4)
    parser.add_argument('--save_epoch', type=int, help='after how many epoch to save the model', default=10)
    parser.add_argument('--device', type=int, help='device number', default=0)
    parser.add_argument('--optimizer', type=str, help='name of the optimizer', default="sgd")
    parser.add_argument('output', type=str, help='path to output network')
    args = parser.parse_args()

    learning_rate = args.lr
    epochs = args.max_epoch
    batch_size = args.batch_size
    num_heads = args.num_heads
    num_per_head = args.num_per_heads
    dim_feedforward = args.dim_feedforward
    num_layers = args.num_layers

    # os.environ["WANDB_DISABLED"] = "true"

    device = "cuda:{}".format(args.device)
    print(device)

    data_dir = args.train_dir
    test_data_dir = args.test_dir
    predicted_csv = os.listdir(data_dir)
    predicted_csv = [pred for pred in predicted_csv if pred.endswith(".csv")]
    predicted_csv_full_path = [os.path.join(data_dir, pred) for pred in predicted_csv]

    test_csv = os.listdir(test_data_dir)
    test_csv = [pred for pred in test_csv if pred.endswith(".csv")]
    test_csv_full_path = [os.path.join(test_data_dir, pred) for pred in test_csv]
    # test if the models of predicted csv and test csv are the same
    if test_csv != predicted_csv:
        print("Val and Test predictions are not the same:\n{}\n{}".format(predicted_csv, test_csv))
        sys.exit(-1)

    val_data = AUTSLDataSet(predicted_csv_full_path, args.train_data, device)
    test_data = AUTSLDataSet(test_csv_full_path, args.test_data, device)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    ensembler = NeuralEnsemblerBERT(14, val_data.num_classes, num_heads, num_per_head, dim_feedforward=dim_feedforward,
                                    num_enc_layers=num_layers)
    # ensembler = NeuralEnsemblerBERTWeighter(14, val_data.num_classes, num_heads, num_per_head,
    #                                         dim_feedforward=dim_feedforward,
    #                                         num_enc_layers=num_layers)
    ensembler = ensembler.to(device)

    if args.optimizer == "sgd":
        optimizer = SGD(ensembler.parameters(), lr=learning_rate, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = Adam(ensembler.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = cross_entropy

    config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_per_head": num_per_head,
        "model_type": str(ensembler),
        "optimizer": optimizer
    }

    wandb.init(project="sensors_2022", entity="mhruz", config=config)

    ensembler = train(ensembler, val_data_loader, epochs, optimizer, criterion, val_data_loader=test_data_loader)

    torch.save(ensembler.state_dict(), os.path.join(wandb.run.dir, args.output))
    wandb.save(args.output)
