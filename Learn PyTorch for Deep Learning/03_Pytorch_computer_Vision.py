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

        self.train_features_batch,self.train_labels_batch=next(iter(self.train_dataloader))
class FashionMNISTModel0(nn.Module):
    def __init__(self,in_shape:int,hidden_units:int,out_shape:int) -> None:
        super().__init__()
        self.layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_shape,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=out_shape),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer_stack(x)
    
    def print_train_time(self,start:float,end:float,device: torch.device=None)->float:
        """Prints difference between start and end time.

        Args:
            start (float): Start time of computation (preferred in timeit format). 
            end (float): End time of computation.
            device ([type], optional): Device that compute is running on. Defaults to None.

        Returns:
            float: time between start and end in seconds (higher is longer).
        """
        total_time = end - start
    
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time
    
    def train_step(self,data:DataLoader,loss_fn,optimizer,accuracy_fn):


        train_loss,train_acc=0,0
        for batch,(X,y) in enumerate(data.train_dataloader):

            y_pred=self(X)

            loss=loss_fn(y_pred,y)
            train_loss+=loss
            train_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(data.train_dataloader)
        train_acc /= len(data.train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(self,data:DataLoader,loss_fn,optimizer,accuracy_fn):

        test_loss,test_acc=0, 0
        self.eval()
        with torch.inference_mode():
            for X,y in data.test_dataloader:

                test_pred=self(X)
                test_loss+=loss_fn(test_pred,y)
                test_acc+=accuracy_fn(y,test_pred.argmax(dim=1))
        
        # Adjust metrics and print out
        test_loss = test_loss / len(data.test_dataloader)
        test_acc /= len(data.test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")    

    def ToTrain(self,data:DataLoader):
        from helper_functions import accuracy_fn
        from timeit import default_timer as timer
        from tqdm.auto import tqdm


        loss_fn=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(self.parameters(),lr=0.1)
        
        train_time_start=timer()
        epochs=5

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n")
            self.train_step(data,loss_fn,optimizer,accuracy_fn)
            self.test_step(data,loss_fn,optimizer,accuracy_fn)

        train_time_stop=timer()
        self.print_train_time(train_time_start,train_time_stop,"CPU")

    def OldMainTrain(self,data:DataLoader):
        from tqdm.auto import tqdm
        from helper_functions import accuracy_fn
        from timeit import default_timer as timer


        loss_fn=nn.CrossEntropyLoss()

        # Train loss: 2.13201 | Test loss: 2.30259, Test acc: 9.98%
        # optimizer=torch.optim.Adam(self.parameters(),lr=0.1)

        # rain loss: 0.55394 | Test loss: 0.40355, Test acc: 85.68%
        optimizer=torch.optim.SGD(self.parameters(),lr=0.1)

        train_time_start_cpu= timer()
        epochs =3 
        for epoch in tqdm(range(epochs)):
            print(f"Epoch : {epoch}\n -------")

            #for testing loss per epochs
            train_loss=0
            for batch, (X,y) in enumerate(data.train_dataloader):
                self.train()
                y_pred=self(X)

                loss=loss_fn(y_pred,y)
                train_loss+=loss

                optimizer.zero_grad()

                loss.backward()
                
                optimizer.step()

                        # Print out how many samples have been seen
                if batch % 400 == 0:
                    print(f"Looked at {batch * len(X)}/{len(data.train_dataloader.dataset)} samples")

            # Divide total train loss by length of train dataloader (average loss per batch per epoch)
            train_loss /= len(data.train_dataloader)

            ##testing
            test_loss,test_acc=0,0

            self.eval()
            with torch.inference_mode():
                for X,y in data.test_dataloader:
                    test_pred=self(X)

                    test_loss+=loss_fn(test_pred,y)

                    test_acc+=accuracy_fn(y,test_pred.argmax(dim=1))
                    
                    # Calculations on test metrics need to happen inside torch.inference_mode()
                    # Divide total test loss by length of test dataloader (per batch)
                test_loss /= len(data.test_dataloader)
                # Divide total accuracy by length of test dataloader (per batch)
                test_acc /= len(data.test_dataloader)

                ## Print out what's happening
                print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

        train_time_end_cpu = timer()
        total_train_time=self.print_train_time(train_time_start_cpu,train_time_end_cpu)


    
class FashionMNISTModelConv(nn.Module):
    # https://poloclub.github.io/cnn-explainer/
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
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
            nn.ReLU(),
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
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        
        )
        print(self.state_dict)

    def forward(self,x:torch.Tensor):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
    
    def print_train_time(self,start:float,end:float,device: torch.device=None)->float:
        """Prints difference between start and end time.

        Args:
            start (float): Start time of computation (preferred in timeit format). 
            end (float): End time of computation.
            device ([type], optional): Device that compute is running on. Defaults to None.

        Returns:
            float: time between start and end in seconds (higher is longer).
        """
        total_time = end - start
    
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    def train_step(self,data:DataLoader,loss_fn,optimizer,accuracy_fn):


        train_loss,train_acc=0,0
        for batch,(X,y) in enumerate(data.train_dataloader):

            y_pred=self(X)

            loss=loss_fn(y_pred,y)
            train_loss+=loss
            train_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(data.train_dataloader)
        train_acc /= len(data.train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(self,data:DataLoader,loss_fn,optimizer,accuracy_fn):

        test_loss,test_acc=0, 0
        self.eval()
        with torch.inference_mode():
            for X,y in data.test_dataloader:

                test_pred=self(X)
                test_loss+=loss_fn(test_pred,y)
                test_acc+=accuracy_fn(y,test_pred.argmax(dim=1))
        
        # Adjust metrics and print out
        test_loss = test_loss / len(data.test_dataloader)
        test_acc /= len(data.test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")    

    def ToTrain(self,data:DataLoader):
        from helper_functions import accuracy_fn
        from timeit import default_timer as timer
        from tqdm.auto import tqdm


        loss_fn=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(self.parameters(),lr=0.1)
        
        train_time_start=timer()
        epochs=5

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n")
            self.train_step(data,loss_fn,optimizer,accuracy_fn)
            self.test_step(data,loss_fn,optimizer,accuracy_fn)

        train_time_stop=timer()
        self.print_train_time(train_time_start,train_time_stop,"CPU")

if __name__=='__main__':
    data=DataFashon()
    p=Plot()
    #p.plotIMGS(data)
    dataloader=DataLoader(data)

    # #testing flatten model
    # flatten_model=nn.Flatten() # all nn modules function as a model (can do a forward pass)

    # x=dataloader.train_features_batch[0]

    # output=flatten_model(x)

    # print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
    # print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")
    model0=FashionMNISTModel0(784,10,len(data.class_names))
    model0.to("cpu")
    print(model0.state_dict)
    model0.ToTrain(dataloader)

    CNNmode0=FashionMNISTModelConv(1,10,len(data.class_names))
    CNNmode0.ToTrain(dataloader)
    
    

