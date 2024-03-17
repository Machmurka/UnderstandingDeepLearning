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
        self.DataLoad()
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
        self.train_features_batch,self.train_labels_batch=next(iter(self.train_dataloader))
        # print(self.train_features_batch.shape,self.train_labels_batch.shape)
class Model0(nn.Module):
    # Based on basic CNN model
    # https://poloclub.github.io/cnn-explainer/
    # ----
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int) -> None:
        super().__init__()

        self.block1=nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block2=nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) 

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,out_features=output_shape)
        )

        print(self.state_dict)

    def test(self,input_shape:int,hidden_units:int,output_shape:int,data:Data):
        self.image=torch.randn(size=(32,1,28,28))
        self.testblok1=nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )


        self.testblok2=nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        print(f'comparing train data shape {data.train_features_batch.shape}\n to data shape after Block1 {self.testblok1(self.image[0]).shape}')
        print(f'comparing Block1 data shape {self.testblok1(self.image[0]).shape}\n to data shape after Block2 {self.testblok2(self.testblok1(self.image[0])).shape}')
        print(f'W_out = (W_in - F + 2P) / S + 1 \nW_out to szerokość (lub wysokość) wyjścia,\nW_in to szerokość (lub wysokość) wejścia,\nF to rozmiar filtra (kernel_size),\nP to padding,\nS to stride.')


class Plots():
    def __init__(self) -> None:
        pass
    
    def PlotSatrtingIMG(self,image:torch.Tensor)->None:
        plt.imshow(image.squeeze(),cmap='gray')
        plt.show()



if __name__=='__main__':
    plot=Plots()
    rawdata=Data(plot)
    model0=Model0(1,3,len(rawdata.class_name))
    model0.test(1,3,len(rawdata.class_name),rawdata)
