import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
class DataMoon:
  def __init__(self)->None:
    pass

  def MakeMoons(self,sample,noise)->None:
    from sklearn.datasets import make_moons
    self.x, self.y=make_moons(n_samples=sample,noise=noise)
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
  def __init__(self) -> None:
    pass

  def PlotMoon(self,data:DataMoon)->None:
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



class MoonModelDropOut(nn.Module):
# Generally, use a small dropout value of 20%-50% of neurons, with 20% providing a good starting point. A probability too low has minimal effect, and a value too high results in under-learning by the network.
# Use a larger network. You are likely to get better performance when Dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.
# Use Dropout on incoming (visible) as well as hidden units. Application of Dropout at each layer of the network has shown good results.
# Use a large learning rate with decay and a large momentum. Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99.
# Constrain the size of network weights. A large learning rate can result in very large network weights. Imposing a constraint on the size of network weights, such as max-norm regularization, with a size of 4 or 5 has been shown to improve results.
  def __init__(self,H_D) -> None:
    super().__init__()
    HIDDEN_LAYER=H_D
    self.LinearStack=nn.Sequential(
       nn.Linear(in_features=2,out_features=HIDDEN_LAYER),
       nn.ReLU(),
       nn.Dropout(p=0.2),
       nn.Linear(in_features=HIDDEN_LAYER,out_features=HIDDEN_LAYER),
       nn.ReLU(),
       nn.Dropout(p=0.2),
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
    epochs=400
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
  data.MakeMoons(1000,0.4)
  p=Plot()
  p.PlotMoon(data)
  Model0=MoonModel(40)
  print(Model0.state_dict)
  Model0.ToTrain(data)
  p.PlotModel(Model0,data)

  Model1=MoonModelDropOut(40)
  print(Model1.state_dict)
  Model1.ToTrain(data)
  p.PlotModel(Model1,data)