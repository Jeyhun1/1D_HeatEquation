from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import scipy.io
import torch
from torch import Tensor, ones, stack, load, nn
from torch.autograd import grad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import pandas as pd


class TeslaDatasetInn(Dataset):
    def __init__(self, pData =  '/content/drive/MyDrive/FrEIA-master/example/tesla_driving_temp_data.csv',ID = -1, device = "cuda:0", normalize = 1, data = "train"):

        # import "tesla_driving_temp_data.csv" dataset
        df = pd.read_csv(pData)
        self.device = device
        
        # Sort the data by date
        #df = df.sort_values(by="date")
        #print(df)
        
        # Interpolate the missing data
        df['outside_temp'] = df['outside_temp'].interpolate()
        df['speed'] = df['speed'].interpolate()
        #print(df) 
        
        #define list of id-values for test data
        values = [16,39,47,52,72,81,88]
        if data != 'all':
          if data == "train":
            #drop any rows that have 7 or 11 in the rebounds column
            df = df[df.drive_id.isin(values) == False]
          elif data == "test" and ID not in values:
            raise ValueError("Pick ID value from following list [16,39,47,52,72,81,88]")
        
        
        # Extract features and labels
        # [power, speed, battery_level, outside_temp] => [battery_temperature]

        df_x = df[["power","speed", "battery_level", "outside_temp"]]
        df_y = df[["battery_temperature"]]
        
        # convert features and labels to tensors
        self.x = torch.tensor(df_x.values).float().to(device)
        self.y = torch.tensor(df_y.values).float().to(device)
        
        # Bounds for the features
        self.lb_x = torch.min(self.x,0).values
        self.ub_x = torch.max(self.x,0).values
        
        # Bounds for the labels
        self.lb_y = torch.min(self.y,0).values
        self.ub_y = torch.max(self.y,0).values
        
        # Normalised features and labels between [-1,1]
        self.x_norm = 2.0*(self.x - self.lb_x)/(self.ub_x- self.lb_x) - 1.0
        self.y_norm = 2.0*(self.y - self.lb_y)/(self.ub_y - self.lb_y) - 1.0

#         # convert features and labels to tensors
#         x = torch.tensor(df_x.values).float().to(device)
#         self.y = torch.tensor(df_y.values).float().to(device)
        
#         # Bounds for the features
#         self.lb_x = torch.min(x,0).values
#         self.ub_x = torch.max(x,0).values
        
#         # Normalised features and labels between [-1,1]
#         self.x_norm = 2.0*(x - self.lb_x)/(self.ub_x- self.lb_x) - 1.0

    def __len__(self):
        return self.x_norm.shape[0]

    
    def __getitem__(self, index):
        xi = self.x_norm[index]
        yi = self.y[index]
        return xi, yi