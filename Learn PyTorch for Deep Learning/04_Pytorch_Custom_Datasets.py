import torch 
import torch.nn as nn

import requests
import zipfile
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms

class SetData():
    def __init__(self) -> None:
      
        # Setup train and testing paths
        self.train_dir = "data\\pizza_steak_sushi\\train"
        self.test_dir = "data\\pizza_steak_sushi\\test"

        random.seed(2137)
        data_dir = Path('C:\\Users\\Jakub Machura\\source\\repos\\UnderstandingDeepLearning\\data') # Create a Path object
        image_path_list = list(data_dir.glob("*/*/*/*.jpg"))  # Call glob on the Path object
        print(image_path_list)        
        random_image_path = random.choice(image_path_list)
        image_class = random_image_path.parent.stem
        img = Image.open(random_image_path)

    def walk_through_dir(self,dir_path):
      import os
      """
      Walks through dir_path returning its contents.
      Args:
        dir_path (str or pathlib.Path): target directory

      Returns:
        A print out of:
          number of subdiretories in dir_path
          number of images (files) in each subdirectory
          name of each subdirectory
      """
      for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")



    def transformData(self):

        self.data_transform=transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Turn the image into a torch.Tensor
            transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        ])

        self.train_data=datasets.ImageFolder(root=self.train_dir,
                                            transform=self.data_transform,
                                            target_transform=None
                                             )
        
        self.test_data=datasets.ImageFolder(root=self.test_dir,
                                            transform=self.data_transform,
                                            target_transform=None)
        
        print(f"Train data { self.train_data}\n Test data{self.test_data}")
        
        # inspect data 
        self.class_names=self.train_data.classes
        
        self.class_dict=self.train_data.class_to_idx

        print(f'class name: {self.class_names}')

        img, label = self.train_data[0][0], self.train_data[0][1]
        print(f"Image tensor:\n{img}")
        print(f"Image shape: {img.shape}")
        print(f"Image datatype: {img.dtype}")
        print(f"Image label: {label}")
        print(f"Label datatype: {type(label)}")

        #plot single img to test
        # img_permutate=img.permute(1,2,0)

        # plt.imshow(img_permutate)
        # plt.show()
    def intoDataLoader(self):
        self.train_dataloader=DataLoader(dataset=self.train_data,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=True)

        self.test_dataloader=DataLoader(dataset=self.test_data,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=False)
        # no need to shuffle testing data

        print(self.train_dataloader,self.test_dataloader)

        img,label=next(iter(self.test_dataloader))
        print(f"shape of img in dataloader: {img.shape}-> [batch_size, color_channels, height, width]")

    
if __name__=="__main__":
    data=SetData()
    # data.ShowImg()
    data.transformData()
    data.intoDataLoader()