import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x.transpose(1, 2))
        out = out.transpose(1, 2)

        out = self.ln1(out)
        out = self.relu(out)
        out = self.drop1(out)
        return out

class DurationPredictor(nn.Module):
    """ Duration Predictor """
    def __init__(self, num_layers, hidden_dim, dropout=0.1):
        super(DurationPredictor, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.append(BasicBlock(hidden_dim, dropout))

        self.layers = nn.ModuleList(layers)

        self.linear_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        out = self.linear_layer(x)
        out = self.relu(out)
        out = out.squeeze(-1)
        return out

class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, hidden_dim, dropout=0.1, num_layer=3, num_class=27):
        super(LengthRegulator, self).__init__()
        self.ebd = nn.Embedding(num_embeddings=num_class, embedding_dim=hidden_dim, padding_idx=0)
        self.duration_predictor = DurationPredictor(num_layer, hidden_dim, dropout)

    def forward(self, x):
        x = self.ebd(x)

        duration_predictor_output = self.duration_predictor(x)
        return duration_predictor_output
