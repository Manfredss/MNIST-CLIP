import torch
from torch import nn
from img_encoding import ImgEncoding
from text_encoding import TextEncoding


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.imgEncoder = ImgEncoding()
        self.textEncoder = TextEncoding()
    def forward(self, img, txt):
        imgEmb = self.imgEncoder(img)
        txtEmb= self.textEncoder(txt)
        return imgEmb @ txtEmb.T
    
if __name__ == '__main__':
    clip = CLIP()
    img = torch.randn(5, 1, 28, 28)
    txt = torch.randint(0, 10, (5,))
    logits = clip(img, txt)
    print(logits.shape)
