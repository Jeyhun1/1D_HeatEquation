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

class TeslaDataset(Dataset):
    def __init__(self, pData = 'tesla_driving_temp_data.csv', device = "cuda:0"):
        # import "tesla_driving_temp_data.csv" dataset
        df = pd.read_csv(pData)
        self.device = device
        
        # Sort the data by date
        df = df.sort_values(by="date")
        #print(df)
        
        # Interpolate the missing data
        df['outside_temp'] = df['outside_temp'].interpolate()
        df['speed'] = df['speed'].interpolate()
        #print(df) 
        
        # Extract features and labels
        # [power, speed, battery_level, outside_temp] => [battery_temperature]

        df_x = df[["power","speed", "battery_level", "outside_temp"]]
        df_y = df[["battery_temperature"]]
        
        self.x = (torch.tensor(df_x.values)).float().to(device)
        self.y = torch.tensor(df_y.values).float().to(device)
        
        self.lb = torch.min(self.x,0).values
        self.ub = torch.max(self.y,0).values
        
        
    def __len__(self):
        return self.x.shape[0]

    
    def __getitem__(self, index):
        xi = self.x[index]
        yi = self.y[index]
        return xi, yi