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
        self.bias , self.weigth = bias, weigth
        self.x=torch.arange(0,1,0.02).unsqueeze(dim=1) # usefull from [50] to [50,1] 
        self.y=self.weigth*self.x+self.bias
    
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

    def Optimizer(self):
        return torch.optim.SGD(params=self.parameters(),lr=0.01)
    #loss function:
    #nn.L1Loss()

d=CreateData(0.3,0.7)
d.CreateTrainTestSplit()
p=Plots()
# p.plot_predictions(d.GetX_train(),d.GetY_train(),d.GetX_test(),d.GetY_test())

#set seed
torch.manual_seed(42)
model0=LinearRegressionModel()

#check the parameters within the nn.module 
# print(list(model0.parameters()))
#other option
print(model0.state_dict())

#Make some predictions with model
with torch.inference_mode():
    y_preds=model0(d.x_test)

# Check the predictions
print(f"Number of testing samples: {len(d.x_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# p.plot_predictions(d.GetX_train(),d.GetY_train(),d.GetX_test(),d.GetY_test(),predictions=y_preds)

loss_fn=nn.L1Loss()


#set number of epochs
epochs = 100

#Create enpty loss list to track values
train_loss_values=[]
test_loss_values=[]
epoch_count=[]

#training
for epoch in range(epochs):
    # Put model in training mode (this is the default state of a model)
    model0.train()

    #1. forward pass on train data using forward()
    y_pred=model0(d.x_train)

    # 2. Calculate the loss
    loss=loss_fn(y_pred,d.y_train)

    # 3. Zero grad of the optimizer
    model0.Optimizer().zero_grad()

    #4. loss backwards
    loss.backward()

    #5. progress the optimizer
    model0.Optimizer().step()

    ##testing
    model0.eval()

    with torch.inference_mode():
        # 1.forward pass on test data
        test_pred=model0(d.x_test)

        # 2. calculate loss on test data
        test_loss = loss_fn(test_pred,d.y_test.type(torch.float))  # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {d.weigth}, bias: {d.bias}")


# making predictions
#1. set the model to eval mode
model0.eval()

# 2. set up interference mode cotext manager
with torch.inference_mode():
    y_pred=model0(d.x_test)
print(f"predicted Y withe eval {y_pred}")
p.plot_predictions(d.GetX_train(),d.GetY_train(),d.GetX_test(),d.GetY_test(),predictions=y_preds)


#5. saving and loading a pytorch model

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model0.state_dict(), 
           f=MODEL_SAVE_PATH)


loaded_model0=LinearRegressionModel()

loaded_model0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model0.eval()

with torch.inference_mode():
    loaded_model_preds=loaded_model0(d.x_test)

print(y_pred==loaded_model_preds)