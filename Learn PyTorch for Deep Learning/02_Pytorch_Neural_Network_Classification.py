import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
import pandas as pd

#make 1000samples 
n_samples=1000
x,y= make_circles(n_samples,noise=0.03,random_state=42)#random_state is like seed
print(f"First 5 x features {x[:5]} \n First 5 y Labels:{y[:5]}")

#make dataframe of circle data
circles=pd.DataFrame({
    "X1":x[:,0],
    "X2" : x[:,1],
    "label":y
})
print(circles.head(10))
print(circles.label.value_counts())

plt.scatter(x=x[:,0],
            y=x[:,1],
            c=y,
            cmap=plt.cm.RdYlBu
            )
plt.show()

#checking in/out shape
print(x.shape,y.shape)

x_sample=x[0]
y_sample=y[0]
print(f"Values for one sample of X: {x_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {x_sample.shape} and the same for y: {y_sample.shape}")

# turning our data as ternsors
x=torch.from_numpy(x).type(torch.float)
y=torch.from_numpy(y).type(torch.float)

#spliting data using train test split from sklearn
from sklearn.model_selection import train_test_split
#test_size=0.2 20% for test and 80% fro train
x_trian,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(len(x_trian),len(x_test))