import torch
from torch import nn
from torch.nn import functional as F

class TextEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=10,
                                embedding_dim=16)
        self.dense1 = nn.Linear(in_features=16,
                                out_features=64)
        self.dense2 = nn.Linear(in_features=64,
                                out_features=16)
        self.fc = nn.Linear(in_features=16,
                            out_features=8)
        self.ln = nn.LayerNorm(normalized_shape=8)

    def forward(self, x):
        out = F.relu(self.dense2(F.relu(self.dense1(self.emb(x)))))
        out = self.ln(self.fc(out))
        return out

if __name__ == '__main__':
    text = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    model = TextEncoding()
    out = model(text)
    # print(out.shape)