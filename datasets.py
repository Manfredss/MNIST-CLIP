import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor, Compose
import matplotlib.pyplot as plt


class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./mnist/',
            train=is_train,
            download=True,)
        self.convert = Compose([PILToTensor(),])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.convert(img)/255.0
        return img, label
    

if __name__ == '__main__':
    dataset = MNIST()
    img, label = dataset[0]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()