import torch 
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class Data():
    def __init__(self,plt) -> None:
        self.TrainData=datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=ToTensor()
        )

        self.TestData=datasets.MNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        )

        image,label = self.TrainData[0]
        print(f"Image shape {image.shape}\n and label {label}")
        self.class_name=self.TrainData.classes
        # plot.PlotSatrtingIMG(image)
    def DataLoad(self)->None:
        from torch.utils.data import DataLoader
        BATCH_SIZE=32 

        #turn into batches
        self.train_dataloader=DataLoader(
            self.TrainData,
            batch_size=BATCH_SIZE
        )

        self.test_dataloader=DataLoader(
            self.TestData,
            batch_size=BATCH_SIZE
        )
        


class Plots():
    def __init__(self) -> None:
        pass
    
    def PlotSatrtingIMG(self,image:torch.Tensor)->None:
        plt.imshow(image.squeeze(),cmap='gray')
        plt.show()



if __name__=='__main__':
    plot=Plots()
    rawdata=Data(plot)
