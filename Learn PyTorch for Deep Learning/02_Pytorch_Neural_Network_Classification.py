import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
import pandas as pd

#  To summarize the above, a model's raw output is referred to as logits.

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
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        print(len(self.x_train),len(self.x_test))
class CircleModel0(nn.Module):
    def __init__(self):
        super().__init__()
        #creating layers 2 nn.Linear layers capable of handling x and y output shape
        self.layer1=nn.Linear(in_features=2, out_features=5)
        self.layer2=nn.Linear(in_features=5,out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x))
    
    def accuracy_fn(self, y_true, y_pred):
        corret=torch.eq(y_true,y_pred).sum().item() # torch.eq() calculates where two tensors are equa
        return(corret/len(y_pred))*100
    
    def ToTrain(self,d:Data)->None:
        #loss fn for binary classification
        loss_fn=nn.BCEWithLogitsLoss() #it uses sigmoid

        #optimizer as a normal SDG
        optimizer=torch.optim.SGD(params=self.parameters(),lr=0.1)

        torch.manual_seed(42)

        epochs=100
        for epoch in range(epochs):
            ##train
            self.train()

            #1 forward pass 
            y_logits=self(d.x_train).squeeze()
            y_pred=torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls

            #2. calculate loss/accuracu
            loss=loss_fn(y_logits,d.y_train)
            acc=model0.accuracy_fn(d.y_train,y_pred)

            #3. optimizer
            optimizer.zero_grad()

            #4 loss backwords
            loss.backward()

            #5. optimizer step
            optimizer.step()

            #testing
            self.eval()
            with torch.inference_mode():
                #1.forward pass
                test_logits=self(d.x_test).squeeze()
                test_pred=torch.round(torch.sigmoid(test_logits))

                #calculate loss
                test_loss=loss_fn(test_logits,d.y_test)
                test_acc=self.accuracy_fn(d.y_test, test_pred)
            # Print out what's happening every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

            #model has 57% acc at best
            # need to plot model decistions
        
class PlotStuff():
    def __init__(self,data:Data) -> None:
        self.d=data
    
    def PlotStartingData(self)->None:
        plt.scatter(x=d.x[:,0],
            y=d.x[:,1],
            c=d.y,
            cmap=plt.cm.RdYlBu
            )
        plt.show()

    def PlotLine(self,model,predictions=None):
        plt.scatter(model.x_train,model.y_train ,c='b' ,s=4)
        plt.scatter(model.x_test,model.y_test , c='g',  s=4)
        if predictions != None: 
            plt.scatter(model.x_test,model.y_pred,c='r', s=4) 
        plt.show()
    def PlotModel(self,model,data)->None:
        import requests
        from pathlib import Path 

        # Download helper functions from Learn PyTorch repo (if not already downloaded)
        if Path("helper_functions.py").is_file():
          print("helper_functions.py already exists, skipping download")
        else:
          print("Downloading helper_functions.py")
          request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
          with open("helper_functions.py", "wb") as f:
            f.write(request.content)

        from helper_functions import plot_predictions, plot_decision_boundary

        #plot using this shit from above
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.title("train")
        plot_decision_boundary(model,data.x_train,data.y_train)
        plt.subplot(1,2,2)
        plt.title("test")
        plot_decision_boundary(model,data.x_test,data.y_test)
        plt.show()

class CircleModelV1(nn.Module):
    def __init__(self,data:Data) -> None:
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=10)
        self.layer2=nn.Linear(in_features=10,out_features=10)
        self.layer3=nn.Linear(in_features=10,out_features=1)
        self.d=data

    def forward(self,x):
        return(self.layer3(self.layer2(self.layer1(x))))

    def accuracy_fn(self, y_true, y_pred):
        corret=torch.eq(y_true,y_pred).sum().item() # torch.eq() calculates where two tensors are equa
        return(corret/len(y_pred))*100

    def ToTrain(self)->None:

        loss_fun=nn.BCEWithLogitsLoss()
        optimizer=torch.optim.SGD(self.parameters(),lr=0.1)

        epochs=1000
        
        for epoch in range(epochs):
            self.train()
            #forward pass
            y_logits=self(d.x_train).squeeze()
            y_pred=torch.round(torch.sigmoid(y_logits)) # logits -> predicition probabilities -> prediction labels

            # 2. calculate loss/acc
            loss=loss_fun(y_logits,d.y_train)
            acc=self.accuracy_fn(d.y_train,y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()            

            self.eval()

            with torch.inference_mode():
                test_logits=self(d.x_test).squeeze()
                test_pred=torch.round(torch.sigmoid(test_logits))

                #calculate loss
                test_loss=loss_fun(test_pred,d.y_test)
                test_acc=self.accuracy_fn(d.y_test,test_pred)
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


class SingleLine():
    def __init__(self) -> None:
        self.CreateData()

    def CreateData(self)->None:
        self.x=torch.arange(0,1,0.01).unsqueeze(dim=1)
        print(self.x)
        self.y=0.3+0.7*self.x
        self.Split()

    def Split(self)->None:
        split=int(len(self.x)*0.8)
        self.x_train , self.y_train = self.x[:split] , self.y[:split]
        self.x_test , self.y_test = self.x[split:] , self.y[split:]
        # print(len(self.x_test))
    def model(self)->None:
        self.model_2= nn.Sequential(
            nn.Linear(in_features=1 , out_features=10),
            nn.Linear(in_features=10,out_features=10),
            nn.Linear(in_features=10 , out_features=1)
        )
         
        loss_fn=nn.L1Loss()
        optimizer=torch.optim.SGD(self.model_2.parameters(),lr=0.1)

        epochs=100
        self.model_2.train()
        for epoch in range(epochs):
            # 1.forward pass
            y_pred=self.model_2(self.x_train)
            
            # 2.calculate loss
            loss = loss_fn(y_pred,self.y_train)

            # 3. optimizer zero grad
            optimizer.zero_grad()

            # 4.loss backwords
            loss.backward()

            # 5.optimizer step
            optimizer.step()

            # 6.testing
            self.model_2.eval()

            with torch.inference_mode():
                test_pred=self.model_2(self.y_test)
                test_loss=loss_fn(self.y_test,test_pred)

                    # Print out what's happening
            if epoch % 100 == 0: 
                print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")
            
            self.model_2.eval()
            with torch.inference_mode():
                self.y_pred = self.model_2(self.x_test)
            

class CircleModelV2(nn.Module):
    def __init__(self,data:Data) -> None:
        super().__init__()
        self.d=data
        self.layer1=nn.Linear(in_features=2,out_features=10)
        self.layer2=nn.Linear(in_features=10, out_features=10)
        self.layer3=nn.Linear(in_features=10,out_features=1)
        #activation function wasn't included before 
        self.relu=nn.ReLU()
    def forward(self,x):
        return(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))))
    
    def accuracy_fn(self, y_true, y_pred):
        corret=torch.eq(y_true,y_pred).sum().item() # torch.eq() calculates where two tensors are equa
        return(corret/len(y_pred))*100
    
    def TrainModel(self)->None:
        loss_fn=nn.BCEWithLogitsLoss()
        optimizer=torch.optim.SGD(self.parameters(), lr=0.1)

        epochs=1000
        for epoch in range(epochs):
            self.train()

            y_logits=self(d.x_train).squeeze()
            y_pred=torch.round(torch.sigmoid(y_logits))

            loss=loss_fn(y_logits,d.y_train)
            acc=self.accuracy_fn(d.y_train,y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            self.eval()

            with torch.inference_mode():
                test_logits=self(d.x_test).squeeze()
                test_pred=torch.round(torch.sigmoid(test_logits))

                test_loss=loss_fn(test_pred,d.y_test)
                test_acc=self.accuracy_fn(d.y_test,test_pred)
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

class BlobData():
    def __init__(self) -> None:
        from sklearn.datasets import make_blobs
        self.NUM_CLASSES = 4
        self.NUM_FEATURES = 2
        self.RANDOM_SEED = 42

        self.x,self.y= make_blobs(n_samples=1000,
            n_features=self.NUM_FEATURES, # X features (x,y point on plot)
            centers=self.NUM_CLASSES, # y labels
            cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
            random_state=self.RANDOM_SEED
        )
    def IntoTensors(self)->None:
        self.x=torch.from_numpy(self.x).type(torch.float)
        self.y=torch.from_numpy(self.y).type(torch.LongTensor) #tensor([3, 2, 2, 1, 1] why as longTensor?
        print(self.x[:5],self.y[:5])
        self.Split()

    def Split(self)->None:
        from sklearn.model_selection import train_test_split
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=0.2,
            random_state=42
        )
    def plotBlob(self):
       
        plt.figure(figsize=(10,7))
        plt.scatter(self.x[:,0],self.x[:,1],c=self.y,cmap=plt.cm.RdYlBu)
        plt.show()

class BlobModel(nn.Module):
    def __init__(self,Data:BlobData,num_hidden_units) -> None:
        super().__init__()
        self.data=Data
        self.linear_layer_stack= nn.Sequential(
            nn.Linear(in_features=self.data.NUM_FEATURES, out_features=num_hidden_units),
            nn.Linear(in_features=num_hidden_units,out_features=num_hidden_units),
            nn.Linear(in_features=num_hidden_units, out_features=self.data.NUM_CLASSES)
        )

    def forward(self,x):
        return self.linear_layer_stack(x)
    
    def accuracy_fn(self, y_true, y_pred):
        corret=torch.eq(y_true,y_pred).sum().item() # torch.eq() calculates where two tensors are equa
        return(corret/len(y_pred))*100

    def TestCheck(self):
        # check if output from network == out of data sample 
        print(self(self.data.x_train)[0], self.data.NUM_CLASSES)


        #  To summarize the above, a model's raw output is referred to as logits.
        y_logits = self(self.data.x_test)
        y_pred_prob=torch.softmax(y_logits,dim=1)
        print(f'logits: {y_logits[:5]}, y_prediction probailities(useage of softmax){y_pred_prob[:5]}')
        # print(torch.sum(y_pred_prob,dim=1))
        print(f'y_prediction probailities{y_pred_prob[0]} and max {torch.argmax(y_pred_prob[0])}')



    def TrainModel(self):
        loss_fn=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(self.parameters(),lr=0.1)

        epochs=1000

        for epoch in range(epochs):
            self.train()
            # go from logits -> 
            y_logits=self(self.data.x_train)
            # prediction probabilities ->         prediction labels
            y_pred=torch.softmax(y_logits, dim=1).argmax(dim=1)
            
            loss=loss_fn(y_logits,self.data.y_train)
            # acc=self.accuracy_fn(self.data.y_train,y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            self.eval()
            with torch.inference_mode():
                test_logit=self(self.data.x_test)
                test_pred=torch.softmax(test_logit,dim=1).argmax(dim=1)
                
                test_loss=loss_fn(test_pred,self.data.y_test)
                # test_acc=self.accuracy_fn(self.data.y_test,test_logit)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}| Test Loss: {test_loss:.5f}")
    
    def Predictions(self):
        self.eval()
        with torch.inference_mode():
            y_logits=self(self.data.x_test)
        print(y_logits[:5])
        print(self.data.y_test[:5],self.data.y_test.shape)
        self.y_preds=torch.softmax(y_logits, dim=1).argmax(dim=1)
        print(self.y_preds[:5],self.y_preds.shape)
        print(f'accuracy of model:{self.accuracy_fn(self.data.y_test,self.y_preds)}')

        from torchmetrics import Accuracy
        torchmetrics_accuracy = Accuracy(task="multiclass",num_classes=int(4))
        print(torchmetrics_accuracy(self.y_preds,self.data.y_test))


        
      


d=Data()
p=PlotStuff(d)
# p.PlotStartingData()
# d.MakeDataframe()

# d.TurnIntoTensor()
# #checking in/out shape
# print(d.x.shape,d.y.shape)

# x_sample=d.x[0]
# y_sample=d.y[0]
# print(f"Values for one sample of X: {x_sample} and the same for y: {y_sample}")
# print(f"Shapes for one sample of X: {x_sample.shape} and the same for y: {y_sample.shape}")

# d.SplitData()
   
# model0=CircleModel0()
# print(model0)

# #we can replace CirceModel0 with
# model_0=nn.Sequential(
#     nn.Linear(in_features=2,out_features=5),
#     nn.Linear(in_features=5,out_features=1)
# )

# untrain_pred=model0(d.x_test)
# print(f"Length of predictions: {len(untrain_pred)}, Shape: {untrain_pred.shape}")
# print(f"Length of test samples: {len(d.y_test)}, Shape: {d.y_test.shape}")
# print(f"\nFirst 10 predictions:\n{untrain_pred[:10]}")
# print(f"\nFirst 10 test labels:\n{d.y_test[:10]}")

# # pred Shape: torch.Size([200, 1])
# # test Shape: torch.Size([200])
# # they differ we fix it later

# model0.ToTrain(d)
# p.PlotModel(model0)
# # Oh wow, it seems like we've found the cause of model's performance issue.
# # It's currently trying to split the red and blue dots using a straight line...
# # That explains the 50% accuracy. Since our data is circular, drawing a straight line can at best cut it down the middle.

# model1=CircleModelV1(d)
# model1.ToTrain()
# p.PlotModel(model1)
# #despite the increase of hidden units it's still single line 

# Singleline=SingleLine()
# Singleline.model()
# # p.PlotLine(Singleline)
# # p.PlotLine(Singleline,Singleline.y_pred)

# model2=CircleModelV2(d)
# model2.TrainModel()
# p.PlotModel(model2)
db=BlobData()
db.IntoTensors()
db.plotBlob()

model3=BlobModel(db,num_hidden_units=8)
print(model3)
# model3.TestCheck()
model3.TrainModel()
model3.Predictions()
p.PlotModel(model3,db)