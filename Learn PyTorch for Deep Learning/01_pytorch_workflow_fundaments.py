import torch
from torch import nn # nn contains all pytorch building blocks
import matplotlib.pyplot as plt

what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

#Create Data for training model
class CreateData:
    def __init__(self,bias,weigth) -> None:
        self.x=torch.arange(0,1,0.02).unsqueeze(dim=1) # usefull from [50] to [50,1] 
        self.y=weigth*self.x+bias
    
    #Create Train ^ Test slit 
    def CreateTrainTestSplit(self) ->None:
        #Training set: The model learns from this data.	~60-80%
        #Testing set: The model gets evaluated on this data to test what it has learned .	~10-20%
        trainSplit=int(0.8*len(self.x))
        self.x_train, self.y_train = self.x[:trainSplit], self.y[:trainSplit]
        self.x_test, self.y_test = self.x[trainSplit:], self.y[trainSplit:]
        print(len(self.x_train),len(self.x_test))
    
    
    #AllGetFucntions
    def GetY(self) -> torch.Tensor:
        return self.y
    def GetX(self) -> torch.Tensor:
        return self.x
    def GetX_train(self) -> torch.Tensor:
        return self.x_train
    def GetY_train(self) -> torch.Tensor:
        return self.y_train
    def GetX_test(self)->torch.Tensor:
        return self.x_test
    def GetY_test(self)->torch.Tensor:
        return self.y_test


    #All Print Fucntions
    def PrintAllPair(self) -> None:
        print(f"Training Data pairs X:{self.GetX()[:,0]} \n and Y: {self.GetY()[:,0]}")

#Creating simple plot class
class Plots:
    def __init__(self) -> None:
        return None
    
    def plot_predictions(slef,train_data,train_labels,test_data,test_labels,predictions=None):
        plt.figure(figsize=(10,7))
        plt.scatter(train_data,train_labels,c="b",s=4, label="Training data")
        plt.scatter(test_data,test_labels, c="g",s=4,label="Testing Data")

        if predictions is not None:
            plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")
        
        plt.legend(prop={"size":14})
        plt.show()



#create Linear Regression model class

# The base class for all neural network modules, all the building blocks for neural networks are subclasses. 
# If you're building a neural network in PyTorch, your models should subclass nn.Module. 
# Requires a forward() method be implemented.
class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # torch.nn.Parameter	
        # Stores tensors that can be used with nn.Module. If requires_grad=True gradients (used for updating model parameters via gradient descent) are calculated automatically, this is often referred to as "autograd".
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float32,requires_grad=True))
        self.bias = nn.Parameter(torch.rand(1,dtype=torch.float32,requires_grad=True))
    
    #Forward defines the computation in the model
    def forward(self,x: torch.Tensor) ->torch.Tensor:
        return self.weights*x+self.bias



d=CreateData(0.3,0.7)
d.CreateTrainTestSplit()
p=Plots()
p.plot_predictions(d.GetX_train(),d.GetY_train(),d.GetX_test(),d.GetY_test())

#set seed
torch.manual_seed(42)
model0=LinearRegressionModel()

#check the parameters within the nn.module 
# print(list(model0.parameters()))
#other option
print(model0.state_dict())