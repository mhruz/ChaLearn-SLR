import torch.nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
import numpy as np


class NeuralEnsemblerBERT(torch.nn.Module):
    def __init__(self, num_models, num_heads, num_per_head):
        super(NeuralEnsemblerBERT, self).__init__()

        encoder_layer = TransformerEncoderLayer(num_per_head * num_heads, num_heads, 512, activation="gelu")
        self.encoder = TransformerEncoder(encoder_layer, 3)

        self.class_token = torch.nn.Parameter(torch.zeros((num_per_head * num_heads)))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((num_models + 1, num_per_head * num_heads)))

        torch.nn.init.uniform_(self.class_token)
        torch.nn.init.xavier_normal_(self.pos_embedding)

    def forward(self, x):

        n, seq, dim = x.shape
        cls_emb = torch.tile(self.class_token, [n, 1, 1])
        pos_emb = torch.tile(self.pos_embedding, [n, 1, 1])
        t = torch.cat((cls_emb, x), 1) + pos_emb
        y = self.encoder(t)

        return y


if __name__ == "__main__":
    num_heads = 14
    num_per_head = 18

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    x = np.random.random((3, num_heads, num_per_head * num_heads))
    x = torch.Tensor(x)
    x = x.to(device)

    ensembler = NeuralEnsemblerBERT(14, num_heads, num_per_head)
    ensembler = ensembler.to(device)

    y = ensembler(x)
    print(y)
