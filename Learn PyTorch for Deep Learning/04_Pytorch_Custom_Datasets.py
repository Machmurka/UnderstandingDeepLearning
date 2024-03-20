import requests
import zipfile
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import os
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from timeit import default_timer as timer 


from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchinfo import summary

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
        self.paths = list(Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
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

        self.train_dir = "data\\pizza_steak_sushi\\train"
        self.test_dir = "data\\pizza_steak_sushi\\test"
        """
            temp placement for of transform
        """
        self.train_transform=transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor()
        ])
    
        self.test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
        ])
    
        self.train_data_custom =ImageFolderCustom(self.train_dir,transform=self.train_transform)
        self.test_data_custom=ImageFolderCustom(self.test_dir,transform=self.test_transforms)
    
        """temp place for testing data"""    
        # Check for equality amongst our custom Dataset and ImageFolder Dataset
        # print((len(self.train_data_custom) == len(data.train_data)) & (len(self.test_data_custom) == len(data.test_data)))
        # print(self.train_data_custom.classes == data.train_data.classes)
        # print(self.train_data_custom.class_to_idx == data.train_data.class_to_idx)

        self.IntoDataLoaders()

    def IntoDataLoaders(self):
        BATCH_SIZE=32
        NUM_WORKERS = os.cpu_count()
        print(f"number of workers avalible {NUM_WORKERS}")
        self.train_dataloader= DataLoader(dataset=self.train_data_custom, # use custom created train Dataset
                                     batch_size=BATCH_SIZE, # how many samples per batch?
                                     num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) 

        self.test_dataloader = DataLoader(dataset=self.test_data_custom, # use custom created test Dataset
                                    batch_size=BATCH_SIZE, 
                                    num_workers=NUM_WORKERS, 
                                    shuffle=False) # don't usually need to shuffle testing data

        img,label=next(iter(self.test_dataloader))
        print(f"shape of custome dataloader img {img.shape}")

class TinnyVGG(nn.Module):
    def __init__(self,in_shape:int,out_shape:int,hidden_units:int) -> None:
        super().__init__()

        self.conv_block1=nn.Sequential(
            nn.Conv2d(in_channels=in_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,out_features=out_shape)
        )
        print(self.state_dict)
    def forward(self,x):
        return(self.classifier(self.conv_block_2(self.conv_block1(x))))
    
    def ShapeCheck(self,data:CustomDataTest):
        img_batch,label_batch=next(iter(data.test_dataloader))

        img_single,label_single=img_batch[0].unsqueeze(dim=0), label_batch[0]
        
        print(f"single img shape {img_single.shape}")

        self.eval()
        with torch.inference_mode():
            y=self(img_single)

        print(f"shape of output raw y\n{y.shape}")
        print(f"output of pred label \n{torch.argmax(torch.softmax(y,dim=1),dim=1)}")
        print(f"actual label \n{label_single}")
    
    def train_step(self,loss_fn,optimizer,data):
        self.train()

        train_loss,train_acc=0, 0

        for batch, (X,y) in enumerate(data.train_dataloader):
            y_logits=self(X)

            loss=loss_fn(y_logits,y)
            train_loss+=loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred_class = y_logits.argmax(dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred_class)

        train_loss=train_loss/len(data.train_dataloader)
        train_acc=train_loss/len(data.train_dataloader)
        return train_loss,train_acc
    
    def test_step(self,loss_fn,data:CustomDataTest):
        
        self.eval()

        test_loss,test_acc=0,0

        with torch.inference_mode():
            for batch, (X,y) in enumerate(data.test_dataloader):
                y_logits=self(X)

                loss=loss_fn(y_logits,y)
                test_loss+=loss.item()

                test_pred_labels = y_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))


        test_loss=test_loss/len(data.test_dataloader)
        test_acc=test_acc/len(data.test_dataloader)
        return test_loss, test_acc

    def Totrain(self,data:CustomDataTest,epochs:int,optimizer,loss_fn:torch.nn.Module):
        
        self.results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    

        for epoch in tqdm(range(epochs)):
            train_loss,train_acc=self.train_step(loss_fn,optimizer,data)

            test_loss,test_acc=self.test_step(loss_fn,data)

                    # 4. Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )
    
            # 5. Update results dictionary
            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            self.results["test_loss"].append(test_loss)
            self.results["test_acc"].append(test_acc)
        
        return self.results
    


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.show()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__=="__main__":
    # data=Food101Data()
    # # # data.ShowImg()
    # data.transformData()
    # data.intoDataLoader()

    data1=CustomDataTest()
    model0=TinnyVGG(3,len(data1.train_data_custom.classes),10)
    # model0.ShapeCheck(data1)
    # summary(model0,input_size=[1,3,64,64])

    model0_optimizer=torch.optim.Adam(params=model0.parameters(),lr=0.001)
    model0_loss_fn=nn.CrossEntropyLoss()
    start_time=timer()

    model0_results=model0.Totrain(data=data1,epochs=15,optimizer=model0_optimizer,loss_fn=model0_loss_fn)
    
    end_time=timer()

    print(f"Total training time: {end_time-start_time:.3f} seconds")

    plot_loss_curves(model0_results)



    