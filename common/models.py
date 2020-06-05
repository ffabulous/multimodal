import torch
from torch import nn


class DTT(nn.Module):
    def __init__(self, seq_len, num_label, k_sizes, num_k, embedding, drop_prob=0.5):
        super(DTT, self).__init__()
        vocab_size, emb_dim = embedding.shape
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb.weight = nn.Parameter(torch.from_numpy(embedding).float())
        self.emb.weight.requires_grad = False
        self.conv = nn.ModuleList()
        for k_size in k_sizes:
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(1, num_k, (k_size, emb_dim), (1, 1), bias=True),
                    nn.BatchNorm2d(num_k),
                    nn.ReLU(),  # NOTE: order of BN, ReLU is controversial
                    nn.MaxPool2d((seq_len - k_size + 1, 1)),
                )
            )
        self.num_k = num_k * len(k_sizes)
        self.fc = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(self.num_k, num_label))

    def forward(self, x):
        x = self.emb(x).unsqueeze(1)  # (N, 1, W, D)
        h = [c(x) for c in self.conv]  # convolve
        h = torch.cat(h, 3).view(-1, self.num_k)  # flatten
        return self.fc(h)

    def predict(self, x):
        return self.forward(x).max(1)  # score, index


class MMVC(nn.Module):
    def __init__(self, r3d, dtt, in_dim, out_dim, drop_prob=0.2):
        super(MMVC, self).__init__()
        self.r3d = r3d
        self.dtt = dtt
        hidden_dim = (in_dim - out_dim) // 2
        self.mlp = nn.Sequential(
            nn.Dropout(drop_prob), nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, xv, xt):
        xv = self.r3d(xv)
        xt = self.dtt(xt)
        return self.mlp(torch.cat([xv, xt], dim=1))

    def predict(self, xv, xt):
        return self.forward(xv, xt).max(1)  # score, index
