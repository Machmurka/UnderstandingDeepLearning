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

import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

class Food101Data():
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


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = self.find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
        
    def find_classes(self,directory:str)->Tuple[list[str],dict[str,int]]:
        classes=sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
        class_to_idx={cls_name: i for i, cls_name in enumerate(classes)}
        
        print(classes,class_to_idx)
        return classes,class_to_idx

class CustomDataTest():
    def __init__(self) -> None:
        """
            temp placement for of transform
        """
        self.train_transform=transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    
        self.test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
        ])
    
        self.train_data_custom =ImageFolderCustom(data.train_dir,transform=self.train_transform)
        self.test_data_custom=ImageFolderCustom(data.test_dir,transform=self.test_transforms)
    
        """temp place for testing data"""    
        # Check for equality amongst our custom Dataset and ImageFolder Dataset
        print((len(self.train_data_custom) == len(data.train_data)) & (len(self.test_data_custom) == len(data.test_data)))
        print(self.train_data_custom.classes == data.train_data.classes)
        print(self.train_data_custom.class_to_idx == data.train_data.class_to_idx)

        self.IntoDataLoaders()

    def IntoDataLoaders(self):
        self.train_dataloader_custom = DataLoader(dataset=self.train_data_custom, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

        self.test_dataloader_custom = DataLoader(dataset=self.test_data_custom, # use custom created test Dataset
                                    batch_size=1, 
                                    num_workers=0, 
                                    shuffle=False) # don't usually need to shuffle testing data

        img,label=next(iter(self.test_dataloader_custom))
        print(f"shape of custome dataloader img {img.shape}")


if __name__=="__main__":
    data=Food101Data()
    # data.ShowImg()
    data.transformData()
    data.intoDataLoader()

    data1=CustomDataTest()
    # data1.find_classes(data.train_dir)
    