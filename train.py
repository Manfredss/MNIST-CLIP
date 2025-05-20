import torch
import os
from torch.nn import functional as F
from clip import CLIP
from datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = MNIST()
model = CLIP().to(device)

try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

# 设置训练参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
iterNum = 50000
batchSize = 64
featureSize = 10
dataloader = DataLoader(dataset,
                       batch_size=batchSize,
                       shuffle=True,)
losses = list()

for epoch in tqdm(range(iterNum), desc='Training', ncols=75):
    while True:
        img, txt = next(iter(dataloader))
        if torch.unique(txt).shape[0] < featureSize:
            continue

        # 挑选出 10 个数字
        target = set()
        idx = list()
        for i in range(batchSize):
            if txt[i].item() in target:
                continue
            target.add(txt[i].item())
            idx.append(i)
            # 所有不同的 label 都被选过了
            if len(target) == featureSize:
                break
        imgs = img[idx]
        labels = txt[idx]
        break
    
    logits = model(imgs.to(device), labels.to(device))

    targets = torch.arange(0, featureSize).to(device)
    lossImg = F.cross_entropy(logits, targets)
    lossTxt = F.cross_entropy(logits.permute(1, 0), targets)
    avgLoss = (lossImg + lossTxt) / 2
    losses.append(avgLoss.detach().numpy())

    optimizer.zero_grad()
    avgLoss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'>> Epoch: {epoch}, Loss: {avgLoss}')
        torch.save(model.state_dict(), '.model.pth')
        os.replace('.model.pth', 'model.pth')

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Time')
plt.savefig('loss.png')
plt.show()