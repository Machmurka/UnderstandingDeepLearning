import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

class DataFashon():
    def __init__(self) -> None:
        self.train_data= datasets.FashionMNIST(
            root='data',
            train=True,
            download=True,
            transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
            target_transform=None # you can transform labels as well
        )

        self.test_data=datasets.FashionMNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        )

        self.image, self.label=self.train_data[0]
        print(f"Image shape ^ tensor:{self.image.shape}\n{self.image}\n Label tensor {self.label}")
        # torch.Size([1, 28, 28] [color_channels=1, height=28, width=28] Having color_channels=1 means the image is grayscale.

        #see classes
        self.class_names = self.train_data.classes
        print(self.class_names)
class Plot():
    def __init__(self):
        pass
    def plotSingleIMG(self,data)->None:
        plt.imshow(data.image.squeeze(),cmap='gray')
        plt.show()
    def plotIMGS(slef,data)->None:
        fig=plt.figure(figsize=(9,9))
        rows,cols=4,4
        for i in range(1,rows*cols+1):
            random_idx=torch.randint(0,len(data.train_data), size=[1]).item()
            img,label=data.train_data[random_idx]
            fig.add_subplot(rows,cols,i)
            plt.imshow(img.squeeze(),cmap='gray')
            plt.title(data.class_names[label])
            plt.axis(False)
        plt.show()
class DataLoader():
    def __init__(self,data:DataFashon) -> None:
        from torch.utils.data import DataLoader
        BATCH_SIZE=32
        
        #turn dataset into batches
        self.train_dataloader= DataLoader(
            data.train_data,
            batch_size=BATCH_SIZE,
            shuffle=True 
        )

        self.test_dataloader=DataLoader(
            data.test_data,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        print(f"Length of train dataloader{len(self.train_dataloader)}\n Length of test dataloader{len(self.test_dataloader)}")
    

if __name__=='__main__':
    data=DataFashon()
    p=Plot()
    #p.plotIMGS(data)
    dataloader=DataLoader(data)