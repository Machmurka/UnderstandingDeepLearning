# import torch 
# import torch.nn as nn
# import matplotlib.pyplot as plt

from sklearn.datasets import make_circles


#make 1000samples 
n_samples=1000
x,y= make_circles(n_samples,noise=0.03,random_state=42)#random_state is like seed
print(f"First 5 x features {x[:5]} \n First 5 y Labels:{y[:5]}")