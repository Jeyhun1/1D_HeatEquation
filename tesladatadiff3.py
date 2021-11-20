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

class TeslaDatasetAll(Dataset):
    def __init__(self, pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv', ID = -1, device = "cuda:0", normalize = 1000, rel_time = False):

        pd.options.mode.chained_assignment = None  # default='warn'
        #pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv'
        #ID=-1
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
        df0['Differentiation'] = df0['deltaTemp']/df0['delta_t']

        #Remove transition points between different drives/dates
        idx=(df0['drive_id'].diff()[df0['drive_id'].diff() != 0].index.values)
        idx=idx-1
        df0 = df0.drop(idx[1:])

        # Remove the NAN values
        remove=(df0['Differentiation'][np.isnan(df0['Differentiation']) == True].index.values)
        df0 = df0.drop(remove)

        #Remove
        idxx = np.where(df0['drive_id'].diff().to_frame()['drive_id']== 1)
        #print(type(idxx))
        idxx = np.asarray(idxx)
        idxx = idxx.reshape(-1)
        idxx = np.append(0, idxx)
        idxx = np.append(idxx,df0['drive_id'].shape[0])

        df0["rel_time"] = np.nan
        
        for k in range(idxx.shape[0]-1):
          #print(k)
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          #print(end_idx)
          df0['rel_time'].iloc[start_idx:end_idx] = df0['time'].iloc[start_idx:end_idx]-df0['time'].iloc[start_idx]

        if rel_time == True:
          print(rel_time)
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature", "rel_time"]]
          df_y = df0[["Differentiation"]]
        else:
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature"]]
          df_y = df0[["Differentiation"]]

        delta_t = df0[["delta_t"]]
        delta_t = torch.tensor(delta_t.values).float().to(device)

        rel_t = df0[["rel_time"]]
        rel_t = torch.tensor(rel_t.values).float().to(device)

        t = df0[["time"]]
        t = torch.tensor(t.values).float()
        
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

        self.lb = lb
        self.ub = ub


    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        #data = self.data.loc[start_idx:end_idx]
        return self.x[start_idx:end_idx],self.y[start_idx:end_idx],self.dt[start_idx:end_idx]
        
    def __len__(self):
        return len(self.batch_indices) - 1

class TeslaDatasetSlice(Dataset):
    def __init__(self,  ID, pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv', device = "cuda:0", normalize = 1000, rel_time = False):

        pd.options.mode.chained_assignment = None  # default='warn'
        #pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv'
        #ID=-1
        # import "tesla_driving_temp_data.csv" dataset
        df = pd.read_csv(pData)
        self.device = device

        if ID == -1:
            df0=df # use all dataset (default)
        else:
            df0=df[df['drive_id'] == ID] # use a slice of dataset based on drive-id

        #df0=df[df['drive_id'] == ID] # use a slice of dataset based on drive-id

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
        df0['Differentiation'] = df0['deltaTemp']/df0['delta_t']

        #Remove transition points between different drives/dates
        idx=(df0['drive_id'].diff()[df0['drive_id'].diff() != 0].index.values)
        idx=idx-1
        df0 = df0.drop(idx[1:])

        # Remove the NAN values
        remove=(df0['Differentiation'][np.isnan(df0['Differentiation']) == True].index.values)
        df0 = df0.drop(remove)

        #Remove
        idxx = np.where(df0['drive_id'].diff().to_frame()['drive_id']== 1)
        #print(type(idxx))
        idxx = np.asarray(idxx)
        idxx = idxx.reshape(-1)
        idxx = np.append(0, idxx)
        idxx = np.append(idxx,df0['drive_id'].shape[0])
        
        df0["rel_time"] = np.nan
                
        for k in range(idxx.shape[0]-1):
          #print(k)
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          #print(end_idx)
          df0['rel_time'].iloc[start_idx:end_idx] = df0['time'].iloc[start_idx:end_idx]-df0['time'].iloc[start_idx]


        if rel_time == True:
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature", "rel_time"]]
          df_y = df0[["Differentiation"]]
        else:
          # Extract features and labels
          df_x = df0[["power","speed", "battery_level", "outside_temp", "battery_temperature"]]
          df_y = df0[["Differentiation"]]

        delta_t = df0[["delta_t"]]
        delta_t = torch.tensor(delta_t.values).float().to(device)

        rel_t = df0[["rel_time"]]
        rel_t = torch.tensor(rel_t.values).float().to(device)

        t = df0[["time"]]
        t = torch.tensor(t.values).float()
        
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

        self.lb = lb
        self.ub = ub

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.dt[index])
    
    def __len__(self):
        return len(self.x)


class TeslaDatasetpinn(Dataset):
    def __init__(self, pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv', ID = -1, device = "cuda:0", normalize = 1000, rel_time = False):

        pd.options.mode.chained_assignment = None  # default='warn'
        #pData = '/content/drive/MyDrive/NeuralSolvers-heat-eqn/examples/Research project/tesla_driving_temp_data.csv'
        #ID=-1
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
        df0['Differentiation'] = df0['deltaTemp']/df0['delta_t']

        #Remove transition points between different drives/dates
        idx=(df0['drive_id'].diff()[df0['drive_id'].diff() != 0].index.values)
        idx=idx-1
        df0 = df0.drop(idx[1:])

        # Remove the NAN values
        remove=(df0['Differentiation'][np.isnan(df0['Differentiation']) == True].index.values)
        df0 = df0.drop(remove)

        #Remove
        idxx = np.where(df0['drive_id'].diff().to_frame()['drive_id']== 1)
        #print(type(idxx))
        idxx = np.asarray(idxx)
        idxx = idxx.reshape(-1)
        idxx = np.append(0, idxx)
        idxx = np.append(idxx,df0['drive_id'].shape[0])

        df0["rel_time"] = np.nan
        
        for k in range(idxx.shape[0]-1):
          #print(k)
          start_idx = idxx[k]
          end_idx = idxx[k+1]
          #print(end_idx)
          df0['rel_time'].iloc[start_idx:end_idx] = df0['time'].iloc[start_idx:end_idx]-df0['time'].iloc[start_idx]

        #df_x = df0[["time"]]
        #df_y = df0[["Differentiation"]]

        df_x = df0[["power","speed", "battery_level", "outside_temp", "rel_time"]]
        df_y = df0[["Differentiation"]]

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
        #print('delta ttttt ', delta_t)

        self.x = df_x_tensor
        self.y = df_y_tensor
        self.dt = delta_t
        self.t = t
        self.batch_indices = idxx
        self.rel_t = rel_t
        self.temp = temp

        self.lb = lb
        self.ub = ub


    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        #data = self.data.loc[start_idx:end_idx]
        return self.x[start_idx:end_idx],self.y[start_idx:end_idx],self.temp[start_idx:end_idx]
        
    def __len__(self):
        return len(self.batch_indices) - 1