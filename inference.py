import random
import torch
import matplotlib.pyplot as plt
from datasets import MNIST
from clip import CLIP
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = MNIST()
model = CLIP().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

''' 预测图片 '''
# 随机挑选一张图片
idx = random.randint(0, len(dataset))
img, txt = dataset[idx]
print('预测图片：', txt)
plt.imshow(img.permute(1, 2, 0))
plt.show()

targets = torch.arange(0, 10)
# 拿到图片，在 0～9 的标签上预测
logits = model(img.unsqueeze(0).to(device), targets.to(device))
print(logits)
print('预测结果：', logits.argmax(-1).item())

''' 找相似图片 '''
similarImgs = list()
similarTxts = list()
# 选取 100 个图片-文本对
for i in range(1, 101):
    similarImg, similarTxt = dataset[i]
    similarImgs.append(similarImg)
    similarTxts.append(similarTxt)

# 拿到 100 张图片的嵌入向量
similarImgsEmb = model.imgEncoder(torch.stack(similarImgs, dim=0).to(device))
# 当前选取的图片的嵌入向量
currImgEmb = model.imgEncoder(img.unsqueeze(0).to(device))
# 计算当前图片与另外 100 张图片的相似度
logits = currImgEmb @ similarImgsEmb.T
# 找到相似度最高的 5 张图片
_, indices = logits[0].topk(5)

plt.figure(figsize=(10, 10))
for idx, imgIdx in enumerate(indices):
    plt.subplot(1, 5, idx + 1)
    plt.imshow(similarImgs[imgIdx].permute(1, 2, 0))
    plt.title(similarTxts[imgIdx])
    plt.axis('off')
plt.show()