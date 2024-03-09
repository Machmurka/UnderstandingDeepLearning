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

class CreateData:
    def __init__(self,bias,weigth) -> None:
        self.x=torch.arange(0,1,0.02).unsqueeze(dim=1) # usefull from [50] to [50,1] 
        self.y=weigth*self.x+bias
    
    def GetY(self) -> torch.Tensor:
        return self.y
    def GetX(self) -> torch.Tensor:
        return self.x
    

Data=CreateData(0.3,0.7)
print(Data.GetY()[:10])