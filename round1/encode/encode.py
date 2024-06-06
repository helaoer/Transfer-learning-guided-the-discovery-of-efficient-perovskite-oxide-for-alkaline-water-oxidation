import numpy as np
import torch
import matplotlib.pyplot as plt    
import pandas as pd
from get_result_encode_r2 import predictor
from data_loader import data_loader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from paras import filepath
from sklearn.preprocessing import LabelEncoder  
class CustomLabelEncoder(LabelEncoder):
    def __init__(self, labels=['-1', 'GC', 'GCE', 'RDE']):         
        self.labels = labels
    def fit(self, y):         
        # Call the fit method of the parent class         
        super().fit(self.labels)
    def transform(self, y):
        # Map the labels to their encoded values, and handle unknown labels by returning 0        
        encoded_labels = super().transform(y)
        return np.array([0 if label not in self.labels else encoded_labels[i] for i, label in enumerate(y)])
    def fit_transform(self, y):         
        # Call the fit_transform method of the parent class, and handle unknown labels by returning 0         
        encoded_labels = super().fit_transform(y)         
        return np.array([0 if label not in self.labels else encoded_labels[i] for i, label in enumerate(y)])

def encode(y_label=False,predict=False,elec_label=1,sub_label="RDE",loading_label=0.51,filepath=filepath,scaler_path=filepath,):
    if os.path.isfile(filepath):
        print(filepath)
        # normalize with all data
        _,_,scaler_x,scaler_y,_,_= data_loader(filepath=scaler_path,y_label=False)
        # set y_label=y_label as substrate and electrolyte are not presented in all data
        # use abo3 index to extract substrate without normalization
        _,_,scaler_x,scaler_y,_,substrate_for_scaler= data_loader(filepath=scaler_path,y_label=y_label,abo3_index=-4)
        # use abo3 index to extract electrolyte without normalization
        _,_,scaler_x,scaler_y,_,electrolyte_for_scaler= data_loader(filepath=scaler_path,y_label=y_label,abo3_index=-7)
        # use abo3 index to extract electrolyte without normalization
        _,_,scaler_x,scaler_y,_,loading_for_scaler= data_loader(filepath=scaler_path,y_label=y_label,abo3_index=-8)

        # y_label filtered data for output
        if predict:
            x,y,_,_,_,abo3= data_loader(filepath=filepath,x_index=[3,-1],scalers=[scaler_x,scaler_y],y_label=y_label)
        else:
            x,y,_,_,_,abo3= data_loader(filepath=filepath,scalers=[scaler_x,scaler_y],y_label=y_label)

        if predict:
            substrate = np.asarray([sub_label]*x.shape[0]).reshape((-1,1))
            electrolyte = np.asarray([elec_label]*x.shape[0]).reshape((-1,1))
            loading = np.asarray([loading_label]*x.shape[0]).reshape((-1,1))
        else:
            _,_,_,_,_,substrate= data_loader(filepath=filepath,scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=-4)
            _,_,_,_,_,electrolyte= data_loader(filepath=filepath,scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=-7)
            _,_,_,_,_,loading= data_loader(filepath=filepath,scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=-8)

        if predict:
            _,_,_,_,_,v_sum= data_loader(filepath=filepath,x_index=[3,-1],scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=6)
            _,_,_,_,_,valences= data_loader(filepath=filepath,x_index=[3,-1],scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=-1)
        else:
            _,_,_,_,_,v_sum= data_loader(filepath=filepath,scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=6)
            _,_,_,_,_,valences= data_loader(filepath=filepath,scalers=[scaler_x,scaler_y],y_label=y_label,abo3_index=-3)

        y = scaler_y.inverse_transform(y)

        # transform substrate to numerical first
        # transform on shape (-1)
        encoder = CustomLabelEncoder() 
        encoder.fit(substrate_for_scaler.reshape((-1,1)))
        substrate_for_scaler = encoder.transform(substrate_for_scaler.reshape((-1)))

        substrate = encoder.transform(substrate.reshape((-1)))
        # normalize
        scaler = StandardScaler()
        scaler.fit(substrate_for_scaler.reshape((-1,1)))
        substrate = scaler.transform(substrate.reshape((-1,1)))           
        substrate = substrate.reshape((-1,1))

        # transform electrolyte
        scaler = StandardScaler()
        scaler.fit(electrolyte_for_scaler.reshape((-1,1)))
        electrolyte = scaler.transform(electrolyte.reshape(-1,1))
        electrolyte = electrolyte.reshape((-1,1))

        # transform mass loading
        scaler = StandardScaler()
        scaler.fit(loading_for_scaler.reshape((-1,1)))
        loading = scaler.transform(loading.reshape(-1,1))
        loading = loading.reshape((-1,1))

        # encode
        pred = predictor()
        x_encode = pred.encode(x)

        columns = ["ABO3","v_sum","valences"]
        for i in range(x_encode.shape[1]):
            columns.append("x_%s"%(i))
        columns += ["loading","electrolyte","substrate"]
        # print(columns)
        if y_label:
            df = pd.DataFrame(np.concatenate((abo3,v_sum,valences,x_encode,loading,electrolyte,substrate,y), axis=1),columns=columns+[y_label[0]])
            df.to_excel("%s.%s.encoded.xlsx"%(filepath[:-5],y_label[0][:5]),index=False)  
            print("Saved to %s.%s.encoded.xlsx"%(filepath[:-5],y_label[0][:5]),)
        else:
            if predict:
                columns = columns+["predict"]
            else:
                columns = columns+["False"]
            df = pd.DataFrame(np.concatenate((abo3,v_sum,valences,x_encode,loading,electrolyte,substrate,y), axis=1),columns=columns)
            df.to_excel("%s.encoded.xlsx"%filepath[:-5],index=False)  
            print("Saved to %s.encoded.xlsx"%filepath[:-5])
