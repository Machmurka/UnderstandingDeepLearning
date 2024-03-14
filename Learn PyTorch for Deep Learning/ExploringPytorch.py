import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
class DataMoon:
  def __init__(self)->None:
    pass

  def MakeMoons(self,sample)->None:
    from sklearn.datasets import make_moons
    self.x, self.y=make_moons(n_samples=sample,noise=0.09)
    self.ToTensor()

  def ToTensor(self)->None:
    self.x=torch.tensor(self.x,dtype=torch.float)
    self.y=torch.tensor(self.y,dtype=torch.float)
    self.Split()

  def Split(self)->None:
    from sklearn.model_selection import train_test_split
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y,test_size=0.2)
    # print(self.x_test.shape,self.x_test[:10],self.x_test.squeeze())

class Plot:
  def __init__(self,Data:DataMoon) -> None:
    self.data=Data

  def PlotCircle(self)->None:
    plt.scatter(data.x[:, 0],data.x[:,1],c=data.y,cmap=plt.cm.RdYlBu)
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

class MoonModel(nn.Module):
    #will add relu later need to chceck if plot module works
  def __init__(self,H_D) -> None:
    super().__init__()
    HIDDEN_LAYER=H_D
    self.LinearStack=nn.Sequential(
       nn.Linear(in_features=2,out_features=HIDDEN_LAYER),
       nn.ReLU(),
       nn.Linear(in_features=HIDDEN_LAYER,out_features=HIDDEN_LAYER),
       nn.ReLU(),
       nn.Linear(in_features=HIDDEN_LAYER,out_features=1)
    )
  def forward(self,x):
    return(self.LinearStack(x))
  
  def TestInOut(self,data:DataMoon):
    self.eval()
    with torch.inference_mode():
      logits=self(data.x_test).squeeze()
      preds=torch.sigmoid(logits)
      labels=torch.round(preds)
      
    # print(labels.dtype,data.y_test.dtype)

  
  def ToTrain(self,data:DataMoon):
    from torchmetrics.classification import BinaryAccuracy
    metric=BinaryAccuracy()
    # optimizer=torch.optim.SGD(self.parameters(),lr=0.01) #bulid basic model then test new optimizers and loss
    optimizer=torch.optim.Adam(self.parameters(),lr=0.01)
    # use CrossEntropyLoss for binary classification, treating it as a two-class problem
    # your target is a float tensor of 0s and 1s, you should use BCEWithLogitsLoss
    loss_fn=nn.BCEWithLogitsLoss() 
    acc_to_plot_train=[]
    acc_to_plot_test=[]
    epochs=300
    for epoch in range(epochs):
      self.train()

      logits=self(data.x_train).squeeze()
      # Sigmoid is used for binary classification methods where we only have 2 classes, 
      # while SoftMax applies to multiclass problems. In fact, the SoftMax function is an extension of the Sigmoid function.
      preds=torch.sigmoid(logits) 


      loss=loss_fn(logits,data.y_train)
      acc=metric(preds,data.y_train)
      acc_to_plot_train.append(acc)
      optimizer.zero_grad()

      loss.backward()
      optimizer.step()

      self.eval()
      with torch.inference_mode():
        test_logits=self(data.x_test).squeeze()
        test_pred=torch.sigmoid(test_logits)

        test_loss=loss_fn(test_logits,data.y_test)
        test_acc=metric(test_pred,data.y_test)
        acc_to_plot_test.append(test_acc)
      if epoch % 20 == 0:
         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
    fig, ax = plt.subplots(nrows=1, ncols=2)
    metric.plot(acc_to_plot_train,ax=ax[0])
    metric.plot(acc_to_plot_test,ax=ax[1])

if __name__=='__main__':
  data=DataMoon()
  data.MakeMoons(1000)
  p=Plot(data)
  #  p.PlotCircle()
  Model0=MoonModel(10)
  print(Model0.state_dict)
  Model0.ToTrain(data)
  p.PlotModel(Model0,data)