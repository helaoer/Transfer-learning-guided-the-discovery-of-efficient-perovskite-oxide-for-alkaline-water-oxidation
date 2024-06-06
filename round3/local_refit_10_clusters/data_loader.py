import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from paras import filepath,no_cluster

def data_loader(filepath=filepath,x_index=[3,-2],y_label="overpotential (mV)",abo3_index=[0],clusters_index=-1,selected_clusters=[i for i in range(no_cluster)],refit=True,scaler=StandardScaler()):
    x = []     
    y = []
    abo3 = []
    clusters = []
    with open(filepath,'rb') as f:
        original_data = pd.read_excel(f)
        feature_names = original_data.columns.values.tolist()

        y_all = original_data[y_label].values
        y_all = np.reshape(y_all, (-1,1))

        x_all = original_data.values
        x_clusters = x_all[:,clusters_index]
        x_abo3 = x_all[:,abo3_index]
        x_features = x_all[:,x_index[0]:x_index[1]]
        feature_names = feature_names[x_index[0]:x_index[1]]

        for i,j,z,c in zip(x_features,y_all,x_abo3,x_clusters):
            abortion = False
            if c not in selected_clusters:                 
                abortion = True
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
                clusters.append(c)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).reshape(-1,1)
        if refit:
            scaler.fit(y)
        y = scaler.transform(y).reshape(-1,1)
        abo3 = np.asarray(abo3).reshape(-1)
        feature_names = np.asarray(feature_names)
        # print("feature names: ", feature_names)
        return x,y,scaler,feature_names,abo3
    
if __name__ == "__main__":     
    x,y,scaler,feature_names,abo3 = data_loader()     
    print(x.shape)         
    print(y.shape)
    
import torch
from torch.utils.data.dataset import Dataset

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