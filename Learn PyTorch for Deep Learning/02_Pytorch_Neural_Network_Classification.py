import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
import pandas as pd

class Data():

    def __init__(self) -> None:
        #make 1000samples 
        n_samples=1000
        self.x,self.y= make_circles(n_samples,noise=0.03,random_state=42)#random_state is like seed
        print(f"First 5 x features {self.x[:5]} \n First 5 y Labels:{self.y[:5]}")
    
    #make dataframe of circle data
    def MakeDataframe(self)->None:
        self.circles=pd.DataFrame({
            "X1":self.x[:,0],
            "X2" : self.x[:,1],
            "label":self.y
        })
        print(self.circles.head(10))
        print(self.circles.label.value_counts())

    def TurnIntoTensor(self)->None:
        # turning our data as ternsors
        self.x=torch.from_numpy(self.x).type(torch.float)
        self.y=torch.from_numpy(self.y).type(torch.float)

    def SplitData(self)->None:
        #spliting data using train test split from sklearn
        from sklearn.model_selection import train_test_split
        #test_size=0.2 20% for test and 80% fro train
        self.x_trian,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        print(len(self.x_trian),len(self.x_test))
class CirceModel0(nn.Module):
    def __init__(self):
        super().__init__()
        #creating layers 2 nn.Linear layers capable of handling x and y output shape
        self.layer1=nn.Linear(in_features=2, out_features=5)
        self.layer2=nn.Linear(in_features=5,out_features=1)

    def forward(self,x):
        return self.layer2(self.layer1(x))
    
    def accuracy_fn(y_true,y_pred)->float:
        corret=torch.eq(y_true,y_pred).sum().item() # torch.eq() calculates where two tensors are equa
        return(corret/len(y_pred))*100
 
    

d=Data()
d.MakeDataframe()
plt.scatter(x=d.x[:,0],
            y=d.x[:,1],
            c=d.y,
            cmap=plt.cm.RdYlBu
            )
plt.show()
d.TurnIntoTensor()
#checking in/out shape
print(d.x.shape,d.y.shape)

x_sample=d.x[0]
y_sample=d.y[0]
print(f"Values for one sample of X: {x_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {x_sample.shape} and the same for y: {y_sample.shape}")

d.SplitData()
   
model0=CirceModel0()
print(model0)

#we can replace CirceModel0 with
model_0=nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features=1)
)

untrain_pred=model0(d.x_test)
print(f"Length of predictions: {len(untrain_pred)}, Shape: {untrain_pred.shape}")
print(f"Length of test samples: {len(d.y_test)}, Shape: {d.y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrain_pred[:10]}")
print(f"\nFirst 10 test labels:\n{d.y_test[:10]}")

# pred Shape: torch.Size([200, 1])
# test Shape: torch.Size([200])
# they differ we fix it later

#loss fn for binary classification
loss_fn=nn.BCEWithLogitsLoss() #it uses sigmoid

#optimizer as a normal SDG
optimizer=torch.optim.SGD(params=model0.parameters(),lr=0.1)


