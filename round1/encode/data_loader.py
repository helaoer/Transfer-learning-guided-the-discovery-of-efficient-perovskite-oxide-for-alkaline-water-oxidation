import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from paras import filepath
# set y_label = False to select encoding data
def data_loader(filepath=filepath,x_index=[3,-8],y_label=["overpotential (mV)"],abo3_index=0,use_scaler=True,scalers=[]):
    x = []
    y = []
    abo3 = []
    feature_names = []
    with open(filepath,'rb') as f:
        original_data = pd.read_excel(f)
        feature_names = original_data.columns.values.tolist()
        if y_label:
            y_all = original_data[y_label].values
            y_all = np.reshape(y_all, (-1,len(y_label)))
        else:
            y_all = np.zeros((original_data.values.shape[0],1))
        x_all = original_data.values
        x_abo3 = x_all[:,abo3_index]
        x_features = x_all[:,x_index[0]:x_index[1]]
        feature_names = feature_names[x_index[0]:x_index[1]]

        for i,j,z in zip(x_features,y_all,x_abo3):
            abortion = False
            for i_ in i:
                if i_ == -1:
                    abortion = True
            for j_ in j:
                if j_ == -1:
                    abortion = True   
            if abortion:
                continue
            else:
                x.append(i)
                y.append(j)
                abo3.append(z)

        x = np.asarray(x)
        y = np.asarray(y)
        abo3 = np.asarray(abo3).reshape(-1,1)
        scaler_x = None
        scaler_y = None
        if use_scaler:
            if len(scalers)!=2:
                scaler_x = StandardScaler()
                scaler_x.fit(x)
                x = scaler_x.transform(x)
                scaler_y = StandardScaler()
                scaler_y.fit(y)
                y = scaler_y.transform(y)
                x = np.asarray(x)
                y = np.asarray(y)
            else:
                scaler_x = scalers[0]
                x = scaler_x.transform(x)
                scaler_y = scalers[1]
                y = scaler_y.transform(y)
                x = np.asarray(x)
                y = np.asarray(y)
                
        return x,y,scaler_x,scaler_y,feature_names,abo3
    
if __name__ == "__main__":
    x,y,scaler_x,scaler_y,feature_names,abo3= data_loader()
    print(x.shape)    
    print(y.shape)

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets

class DatasetConverter(Dataset):

    def __init__(self, x, y, to_cuda=True):
        self.x = x.copy()
        self.y = y.copy()

        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)

        if to_cuda:
            self.x = torch.from_numpy(self.x).type(torch.FloatTensor).cuda()
            self.y = torch.from_numpy(self.y).type(torch.FloatTensor).cuda()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)