import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import  fetch_california_housing, load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torchvision 
import torchvision.transforms as transforms
class RegressionDataset(Dataset):
    """Sin dataset."""

    def __init__(self, num_samples, args = None,settype="sin"):
        """
        Args:
            num_samples (int): Number of samples to generate.
        """
        self.settype = settype
        if settype == "sin":
            self.num_samples = num_samples
            self.x = torch.rand(num_samples) * 2 * np.pi
            self.y = torch.sin(self.x)
        
        elif settype == "cos":
            self.num_samples = num_samples
            self.x = torch.rand(num_samples,1) * 2 * np.pi
            self.y = torch.cos(self.x)

        elif settype == "relu":
            self.num_samples = num_samples
            self.x = torch.rand(num_samples) * 10 - 5
            self.y = torch.nn.functional.relu(self.x)
            
        elif settype == "exp":
            self.num_samples = num_samples
            self.x = torch.rand(num_samples) 
            self.y = torch.exp(self.x)
        
        elif settype == "reversed":
            self.num_samples = num_samples
            self.x = torch.randint(low=1, high=100, size=(num_samples, 5)).float()
            self.y = torch.flip(self.x, dims=[1])
            
        elif settype == "sin_cos":
            self.x = torch.rand(num_samples) * 2 * np.pi
            self.y = torch.sin(self.x) + torch.cos(self.x)
        
        elif settype == "exp_sin":
            self.x = (torch.rand(num_samples) * 2* np.pi) - np.pi
            self.y = torch.exp(self.x) + torch.sin(self.x)
        
        elif settype == "exp_rec":
            self.x = torch.rand(num_samples) * 2* np.pi + 1
            self.y = torch.exp(self.x) + torch.reciprocal(torch.exp(self.x))

        elif settype == "log_sin":
            self.x = torch.rand(num_samples) *1* np.pi
            self.y = torch.log(self.x) + torch.sin(self.x)

        elif settype == "exp_log":
            # x should be more than 1, less than pi
            self.x = torch.rand(num_samples) * np.pi 
            self.x = torch.clamp(self.x, min=1, max=np.pi)
            self.y = torch.exp(self.x) + torch.log(self.x)
        
        elif settype == "exp_sqrt":
            self.x = torch.rand(num_samples) * 2* np.pi
            self.y = torch.exp(-self.x) + torch.sqrt(self.x)
        
        elif settype == "rexp_sin":
            self.x = torch.rand(num_samples) * 2* np.pi - np.pi
            self.y = torch.exp(-self.x) + torch.sin(self.x)


        elif settype == "mixed":
            self.num_samples = num_samples
            self.x = torch.rand(num_samples) * 2 * np.pi
            # 가중치에 따라서 어떻게 변하는가?
            self.y = (2* torch.sin(self.x)) + (1* torch.cos(self.x))

        elif settype == "modularsum":
            self.num_samples = num_samples
            self.x = torch.randint(low=1, high=100, size=(num_samples, 2)).float()
            
            mode = 5
            self.y = torch.div((torch.sum(self.x, dim=1) % mode).float(), 10)
            
            # one-hot encoding of x and y
            # self.x = torch.nn.functional.one_hot(self.x.long()).float()
            # self.y = torch.nn.functional.one_hot(self.y.long()).float()
            self.in_length = 2
            self.cls_num = 1

            if args.onehot == 1:
                self.x = torch.nn.functional.one_hot(self.x.long()).float()
                self.y = torch.nn.functional.one_hot(self.y.long()).float()
                self.in_length = 100
                self.cls_num = 1
            
            

            
        
        elif settype == "california":
            california = fetch_california_housing()
            data, target = california.data, california.target
            
            #normalize
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
            
            
            self.x = torch.Tensor(data)
            self.y = torch.Tensor(target)


            
        elif settype == "5digit":
            # 5 digits addition
            self.num_samples = num_samples
            self.x = torch.randint(low=0, high=9, size=(num_samples, 10)).float()
            self.x_1 = self.x[:, :5]
            self.x_2 = self.x[:, 5:]
            self.x_1 = self._decimal_conversion(self.x_1)
            self.x_2 = self._decimal_conversion(self.x_2)
            self.y = self.x_1 + self.x_2
            
         
        else:
            raise ValueError("Settype must be specified, check datafeed.py.")
        
        
    def _decimal_conversion(self,x):
        
        res = 0
        for i in range(len(x)):
            res += x[i] * (10**i)
        
        return res
        
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # item_x = self.x[idx].unsqueeze(dim=0)
        if self.settype == "modularsum" or self.settype == "boston":
            item_x = self.x[idx].unsqueeze(dim=-1)
        item_x = self.x[idx].unsqueeze(dim=0)
        item_y = self.y[idx].unsqueeze(dim=0)
        return item_x, item_y




class ClassificationDataset:
    def __init__(self, settype="iris"):
        self.settype = settype
        self.cls_num = None
        
        if settype == "iris":
            iris = load_iris()
            data, target = iris.data, iris.target
            #normalize
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
            self.x = torch.Tensor(data)
            self.y = torch.Tensor(target)
            self.cls_num = max(target) + 1
            # one-hot y
            self.y = torch.nn.functional.one_hot(self.y.long()).float()
            self.in_length = 4
            
            
        elif settype == "mnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            self.x = dataset.data.float()
            self.y = dataset.targets
            self.cls_num = max(self.y) + 1
            # flatten each image
            self.x = self.x.view(-1, 28*28)
            self.in_length = 28*28
            self.y = torch.nn.functional.one_hot(self.y.long()).float()
            
        elif settype == "spiral":
            # For spiral regression dataset
            # Load the spiral dataset from the CSV file
            spiral_data = pd.read_csv("./misc/spiral.csv")

            # Extract the x, y, and label columns
            x = spiral_data["x"].values
            y = spiral_data["y"].values
            label = spiral_data["label"].values

            # Convert the data to torch tensors
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            label = torch.Tensor(label)

            # Combine the x and y values into a single tensor
            self.x = torch.stack((x, y), dim=1)
            self.y = label.long()
            # one-hot y
            self.y = torch.nn.functional.one_hot(self.y.long()).float()
            self.cls_num = int(max(label).item()) +1
            self.in_length = 2

        elif settype == "cifar10":
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            # transform goes here
            self.x = torch.Tensor(dataset.data)
            self.x = self.x.data.permute(0,3,1,2)
            self.x = transforms.functional.normalize(self.x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.x = transforms.functional.rgb_to_grayscale(self.x)
            self.x = transforms.functional.resize(self.x, [28,28])
            # self.x = self.x.view(-1, 28*28)
            self.y = torch.Tensor(dataset.targets).float()
            self.cls_num = max(self.y) + 1
            # flatten each image
            self.in_length = 28*28
            self.cls_num = 10
            self.y = torch.nn.functional.one_hot(self.y.long()).float()
            
        else:
            raise ValueError("Settype must be specified, check datafeed.py.")
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # think why did we used unsqueeze here. 
        # was it necessary for batch-wise processing?
        item_x = self.x[idx]
        item_y = self.y[idx]
        return item_x, item_y