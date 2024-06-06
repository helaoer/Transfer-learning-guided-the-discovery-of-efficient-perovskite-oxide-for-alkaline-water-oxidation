import os
import numpy as np
import pandas as pd
import pickle
from data_loader import data_loader
from paras import no_cluster,cluster_centres
from train import single_regressor,multi_regressor

regressors = []
for i in range(no_cluster):
    with open('./models/clf_best_cluster_%s.pkl'%(i),"rb") as f:
        regressor = pickle.load(f)
        regressors.append(regressor)

def get_distance(data1,data2):
    return np.linalg.norm(data1 - data2, axis=0).reshape((-1))

x,y,scaler,feature_names,abo3 = data_loader(abo3_index=[0,1,2]) 
y = scaler.inverse_transform(y)
df = []
for regressor in regressors:
    dist = [get_distance(x_[:-3],regressor.centre) for x_ in x]
    dist = np.array(dist).reshape((-1))
    df.append(
        regressor.predict(x).reshape((-1))/(dist+1)
    )

columns = ["ABO3","v_sum","valences"]
for i in range(no_cluster):
    columns.append("Local %s"%(i))
columns.append("True")
df = np.asarray(df)
df = pd.DataFrame(np.concatenate((abo3.reshape((-1,3)),df.T,y.reshape((-1,1))),axis=1),columns=columns)
df.to_excel("local_results.xlsx",index=False)
