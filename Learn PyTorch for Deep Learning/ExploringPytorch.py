import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
class DataMoon:
   def __init__(self)->None:
        pass
   def MakeMoons(self)->None:
        from sklearn.datasets import make_circles
        self.x, self.y=make_circles(n_samples=100,noise=0.1)
        print(self.x,self.y)

class Plot:
    def __init__(self) -> None:
        pass
    def PlotCircle(self,data:DataMoon)->None:
        plt.scatter(data.x[:, 1])
            
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
    
   def __init__(self) -> None:
      super().__init__()
      pass
   
if __name__=='__main__':
   data=DataMoon()
   data.MakeMoons()