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


class TeslaDatasetNo(Dataset):
    def __init__(self, pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv',ID = -1, device = "cuda:0", normalize = 1, data = "train", rel_time = False, diff = "fwd_diff"):
        """
        Constructor for the dataset for the Neural Operator model

        Args: 
            pData: path for the tesla dataset
            ID: is the ID number of the corresponding drive in the dataset (default ID=-1 corresponds to all data)
            device: represents the device on which the computations will take place ("cuda:0" or "cpu")
            normalize: a coefficient to normalize the low values of the differential operator
            data: represents which type of data is considered. "all" is all data, "train" is the training data, "test" is the test data
            rel_time: represents whether or not to include relative time as input parameter
            diff: represents the method to calculate the time derivative ("fwd_diff" for forward difference, "central_diff" for central differences)
        """   

        pd.options.mode.chained_assignment = None  # default='warn'

        # import "tesla_driving_temp_data.csv" dataset
        df = pd.read_csv(pData)
        self.device = device

        if ID == -1:
            df0=df # use all dataset (default)
        else:
            df0=df[df['drive_id'] == ID] # use a slice of dataset based on drive-id

        # Interpolate the missing data
        print(df0.shape)
        df0['outside_temp'] = df['outside_temp'].interpolate()
        df0['speed'] = df['speed'].interpolate()

        # convert date string to datetime format
        df0['date']= pd.to_datetime(df0['date'])

        # subtract all the data from the initial condition date
        df0['time'] = df0['date'] - df0['date'].iloc[0]

        # change the time fromat to seconds
        df0['time'] = df0['time'][1:]/ np.timedelta64(1, 's')
        df0['time'].iloc[0] = 0

        # calculate change in time between current and next time step and save it in column delta t
        df0['delta_t'] = df0['time'].diff(periods=-1)*(-1)

        #interpolate the missing delta_t
        df0['delta_t'] = df0['delta_t'].interpolate()

        # Compute deltaTemp between current and next time step and add an additional column to dataset
        df0['deltaTemp'] = df0['battery_temperature'].diff(periods=-1)*(-1)
        #Interpolate deltaTemp
        df0['deltaTemp'] = df0['deltaTemp'].interpolate()
        
        # Calculate the differential operator
        df0['fwd_diff'] = df0['deltaTemp']/df0['delta_t']

        #Remove transition points between different drives/dates
        idx=(df0['drive_id'].diff()[df0['drive_id'].diff() != 0].index.values)
        idx=idx-1
        df0 = df0.drop(idx[1:])

        # Remove the NAN values
        remove=(df0['fwd_diff'][np.isnan(df0['fwd_diff']) == True].index.values)
        df0 = df0.drop(remove)

        def find_indices(df0):
          # indices of transition points between drives
          idxx = np.where(df0['drive_id'].diff().to_frame()['drive_id'] >= 1)
          #print(type(idxx))
          idxx = np.asarray(idxx)
          idxx = idxx.reshape(-1)
          idxx = np.append(0, idxx)
          idxx = np.append(idxx,df0['drive_id'].shape[0])
          return idxx
        
        idxx = find_indices(df0)
        
        # create new column corresponding to relative time
        df0["rel_time"] = np.nan
                
        for k in range(idxx.shape[0]-1):
          #print(k)
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          #print(end_idx)
          df0['rel_time'].iloc[start_idx:end_idx] = df0['time'].iloc[start_idx:end_idx]-df0['time'].iloc[start_idx]
        
        def diff_central(x, y):
          x0 = x[:-2]
          x1 = x[1:-1]
          x2 = x[2:]
          y0 = y[:-2]
          y1 = y[1:-1]
          y2 = y[2:]
          f = (x2 - x1)/(x2 - x0)
          #print('f', (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0))
          return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)
         
        df0["central_diff"] = np.nan

        for k in range(idxx.shape[0]-1):
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          xx = df0['rel_time'].iloc[start_idx:end_idx].values
          yy = df0['battery_temperature'].iloc[start_idx:end_idx].values
          df0['central_diff'].iloc[start_idx:end_idx-2] = diff_central(xx,yy)

        df0['central_diff'] = df0['central_diff'].interpolate()
        
        df0_orig = df0

        #define list of id-values for test data
        values = [16,39,47,52,72,81,88]
        if data != 'all':
          if data == "train":
            #drop any rows that have 7 or 11 in the rebounds column
            df0 = df0[df0.drive_id.isin(values) == False]
          elif data == "test" and ID not in values:
            raise ValueError("Pick ID value from following list [16,39,47,52,72,81,88]")
        
        idxx = find_indices(df0)

        # Extract features and labels
        df_x = df0[["power","speed", "battery_level", "outside_temp"]]
        df_y = df0[["battery_temperature"]]
        
        if rel_time == True:
          #print(rel_time)
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature", "rel_time"]]
          #df_y = df0[["fwd_diff"]]
        else:
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature"]]
          #df_y = df0[["fwd_diff"]]
        
        if diff == "fwd_diff":
          #print(rel_time)
          # Extract features and labels
          #df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature", "rel_time"]]
          df_y = df0[["fwd_diff"]]
            
        if diff == "central_diff":
          # Extract features and labels
          #df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature"]]
          df_y = df0[["central_diff"]]


        delta_t = df0[["delta_t"]]
        delta_t = torch.tensor(delta_t.values).float().to(device)

        rel_t = df0[["rel_time"]]
        rel_t = torch.tensor(rel_t.values).float().to(device)

        t = df0[["time"]]
        t = torch.tensor(t.values).float()

        temp = df0[["battery_temperature"]]
        temp = torch.tensor(temp.values).float()
        
        # Normalisation = 1000
        df_x_tensor = torch.tensor(df_x.values).float()
        df_y_tensor = torch.tensor(df_y.values).float()*normalize

        # Bounds
        lb = torch.min(df_x_tensor,0).values.numpy()
        ub = torch.max(df_x_tensor,0).values.numpy()
        lb[3]=df[['outside_temp']].min()
        ub[3]=df[['outside_temp']].max()

        self.x = df_x_tensor.to(device)
        self.y = df_y_tensor.to(device)
        self.dt = delta_t
        self.t = t
        self.batch_indices = idxx
        self.rel_t=rel_t
        self.df0_orig = df0_orig
        self.df0 = df0
        self.temp = temp

        self.lb = lb
        self.ub = ub

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)



class TeslaDatasetNoStb(Dataset):
    def __init__(self, pData = '/tesla_driving_temp_data.csv',ID = -1, device = "cuda:0", normalize = 1, data = "train", rel_time = False, diff = "fwd_diff"):       
        """
        Constructor for the dataset for the Neural Operator (time stability) model

        Args: 
            pData: path for the tesla dataset
            ID: is the ID number of the corresponding drive in the dataset (default ID=-1 corresponds to all data)
            device: represents the device on which the computations will take place ("cuda:0" or "cpu")
            normalize: a coefficient to normalize the low values of the differential operator
            data: represents which type of data is considered. "all" is all data, "train" is the training data, "test" is the test data
            rel_time: represents whether or not to include relative time as input parameter
            diff: represents the method to calculate the time derivative ("fwd_diff" for forward difference, "central_diff" for central differences)
        """   
        pd.options.mode.chained_assignment = None  # default='warn'
        # import "tesla_driving_temp_data.csv" dataset
        df = pd.read_csv(pData)
        self.device = device

        if ID == -1:
            df0=df # use all dataset (default)
        else:
            df0=df[df['drive_id'] == ID] # use a slice of dataset based on drive-id

        # Interpolate the missing data
        print(df0.shape)
        df0['outside_temp'] = df['outside_temp'].interpolate()
        df0['speed'] = df['speed'].interpolate()

        # convert date string to datetime format
        df0['date']= pd.to_datetime(df0['date'])

        # subtract all the data from the initial condition date
        df0['time'] = df0['date'] - df0['date'].iloc[0]

        # change the time fromat to seconds
        df0['time'] = df0['time'][1:]/ np.timedelta64(1, 's')
        df0['time'].iloc[0] = 0

        # calculate change in time between current and next time step and save it in column delta t
        df0['delta_t'] = df0['time'].diff(periods=-1)*(-1)

        #interpolate the missing delta_t
        df0['delta_t'] = df0['delta_t'].interpolate()

        # Compute deltaTemp between current and next time step and add an additional column to dataset
        df0['deltaTemp'] = df0['battery_temperature'].diff(periods=-1)*(-1)
        #Interpolate deltaTemp
        df0['deltaTemp'] = df0['deltaTemp'].interpolate()
        
        # Calculate the differential operator
        df0['fwd_diff'] = df0['deltaTemp']/df0['delta_t']

        #Remove transition points between different drives/dates
        idx=(df0['drive_id'].diff()[df0['drive_id'].diff() != 0].index.values)
        idx=idx-1
        df0 = df0.drop(idx[1:])

        # Remove the NAN values
        remove=(df0['fwd_diff'][np.isnan(df0['fwd_diff']) == True].index.values)
        df0 = df0.drop(remove)

        def find_indices(df0):
          # indices of transition points between drives
          idxx = np.where(df0['drive_id'].diff().to_frame()['drive_id'] >= 1)
          #print(type(idxx))
          idxx = np.asarray(idxx)
          idxx = idxx.reshape(-1)
          idxx = np.append(0, idxx)
          idxx = np.append(idxx,df0['drive_id'].shape[0])
          return idxx
        
        idxx = find_indices(df0)
        
        # create new column corresponding to relative time
        df0["rel_time"] = np.nan
                
        for k in range(idxx.shape[0]-1):
          #print(k)
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          #print(end_idx)
          df0['rel_time'].iloc[start_idx:end_idx] = df0['time'].iloc[start_idx:end_idx]-df0['time'].iloc[start_idx]
        
        def diff_central(x, y):
          x0 = x[:-2]
          #print('x0',x0)
          x1 = x[1:-1]
          x2 = x[2:]
          y0 = y[:-2]
          y1 = y[1:-1]
          y2 = y[2:]
          f = (x2 - x1)/(x2 - x0)
          #print('f', (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0))
          return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)
         
        df0["central_diff"] = np.nan

        for k in range(idxx.shape[0]-1):
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          xx = df0['rel_time'].iloc[start_idx:end_idx].values
          yy = df0['battery_temperature'].iloc[start_idx:end_idx].values
          df0['central_diff'].iloc[start_idx:end_idx-2] = diff_central(xx,yy)

        df0['central_diff'] = df0['central_diff'].interpolate()
        
        df0_orig = df0

        #define list of id-values for test data
        values = [16,39,47,52,72,81,88]
        if data != 'all':
          if data == "train":
            #drop any rows that have 7 or 11 in the rebounds column
            df0 = df0[df0.drive_id.isin(values) == False]
          elif data == "test" and ID not in values:
            raise ValueError("Pick ID value from following list [16,39,47,52,72,81,88]")

        
        idxx = find_indices(df0)

        # Extract features and labels
        # df_x = df0[["power","speed", "battery_level", "outside_temp"]]
        # df_y = df0[["battery_temperature"]]
        
        if rel_time == True:
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature", "rel_time"]]

        else:
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature"]]
        
        if diff == "fwd_diff":
          # Extract features and labels
          df_y = df0[["fwd_diff"]]
            
        if diff == "central_diff":
          # Extract features and labels
          df_y = df0[["central_diff"]]


        delta_t = df0[["delta_t"]]
        delta_t = torch.tensor(delta_t.values).float().to(device)

        rel_t = df0[["rel_time"]]
        rel_t = torch.tensor(rel_t.values).float().to(device)

        t = df0[["time"]]
        t = torch.tensor(t.values).float()

        temp = df0[["battery_temperature"]]
        temp = torch.tensor(temp.values).float()
        
        # Normalisation = 1000
        df_x_tensor = torch.tensor(df_x.values).float()
        df_y_tensor = torch.tensor(df_y.values).float()*normalize

        # Bounds
        lb = torch.min(df_x_tensor,0).values.numpy()
        ub = torch.max(df_x_tensor,0).values.numpy()
        lb[3]=df[['outside_temp']].min()
        ub[3]=df[['outside_temp']].max()

        self.x = df_x_tensor
        self.y = df_y_tensor
        self.dt = delta_t
        self.t = t
        self.batch_indices = idxx
        self.rel_t=rel_t
        self.df0_orig = df0_orig
        self.df0 = df0
        self.temp = temp

        self.lb = lb
        self.ub = ub

    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        return self.x[start_idx:end_idx],self.y[start_idx:end_idx],self.dt[start_idx:end_idx], self.rel_t[start_idx:end_idx]

    def __len__(self):
        return len(self.batch_indices) - 1