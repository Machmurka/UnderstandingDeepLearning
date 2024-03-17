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

        # epochs 10 , batchsize=32 , nn.Dropout(p=0.5)
        # Train loss: 0.08363 | Train accuracy: 97.39%
        # Test loss: 0.10945 | Test accuracy: 96.61%


        # epochs 10 , batchsize=32 , no regu adde
        # Train loss: 0.07794 | Train accuracy: 97.62%
        # Test loss: 0.09471 | Test accuracy: 97.20%
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
    def forward(self,x):
        return(self.classifier(self.block2(self.block1(x))))
    
    def print_train_time(self,start:float,end:float)->float:
        total_time = end - start
    
        print(f"Train time : {total_time:.3f} seconds")
        return total_time

    def train_step(self,data:Data,loss_fn,optimizer,accuracy_fn):
        train_loss,train_acc=0,0
        for batch,(X,y) in enumerate(data.train_dataloader):
            y_pred=self(X)

            loss=loss_fn(y_pred,y)
            train_loss+=loss
            train_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss/=len(data.train_dataloader)
        train_acc/=len(data.train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    
    def test_step(self,data:Data,loss_fn,accuracy_fn):
        
        test_loss,test_acc=0, 0
        self.eval()
        with torch.inference_mode():
            for X,y in data.test_dataloader:

                test_pred=self(X)
                test_loss+=loss_fn(test_pred,y)
                test_acc+=accuracy_fn(y,test_pred.argmax(dim=1))
        
        # Adjust metics and print out
        test_loss = test_loss / len(data.test_dataloader)
        test_acc /= len(data.test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")    

    def ToTrain(self,data:Data):
        from helper_functions import accuracy_fn
        from timeit import default_timer as timer
        from tqdm.auto import tqdm

        loss_fn=nn.CrossEntropyLoss()
        
        # test for furhter optimalization
        optimizer=torch.optim.SGD(self.parameters(),lr=0.1)

        train_time_start=timer()

        epochs=10

        for epoch in tqdm(range(epochs)):
            print(f"\nEpoch: {epoch}\n")
            self.train_step(data,loss_fn,optimizer,accuracy_fn)
            self.test_step(data,loss_fn,accuracy_fn)
            
        train_time_stop=timer()
        self.print_train_time(train_time_start,train_time_stop)
    
    def Matrixreport(self,data:Data):
        from sklearn.metrics import confusion_matrix, classification_report
        self.eval()
        y_preds=[]
        true_pred=[]
        with torch.inference_mode():
            for X,y in data.test_dataloader:
                y_logits=self(X)
                y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
                y_preds.append(y_pred)
                true_pred.append(y)
        y_preds,true_pred=torch.cat(y_preds),torch.cat(true_pred)
        print(classification_report(true_pred,y_preds))
        cm=confusion_matrix(true_pred,y_preds)
        print(cm)

        
    def IOtest(self,input_shape:int,hidden_units:int,output_shape:int,data:Data):
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
    model0.ToTrain(rawdata)
    # model0.IOtest(1,3,len(rawdata.class_name),rawdata)
    # czy możemy to narysować jak podział daty w zeszłym notebook


