import os
import numpy as np
import pandas as pd
import pickle
from data_loader import data_loader
from train import single_regressor,multi_regressor


def get_distance(data1,data2):
    return np.linalg.norm(data1 - data2, axis=0).reshape((-1))

def encode_local(
    files,
    outfiles,
    model_paths = [
        "../local_refit_crystal_clusters/models/",
        "../local_refit_1_clusters/models/",
        "../local_refit_5_clusters/models/",
        "../local_refit_6_clusters/models/",
        "../local_refit_13_clusters/models/",
        "../local_refit_15_clusters/models/",
        "../local_refit_18_clusters/models/",
    ],
    y_label="overpotential (mV)",
    x_index=[3,-2],
    abo3_index=[0,1,2],
    columns = ["ABO3","v_sum","valences"],
):
    regressors = []
    for model_path in model_paths:
        for i in range(1000):
            model_file = model_path + "clf_best_cluster_%s.pkl"%(i)
            if os.path.isfile(model_file):
                with open(model_file,"rb") as f:
                    regressor = pickle.load(f)
                    regressors.append(regressor)
    if os.path.isfile(files):
        print(files)
        x,y,scaler,feature_names,abo3 = data_loader(filepath=files,abo3_index=abo3_index,y_label=y_label,x_index=x_index) 
        y = scaler.inverse_transform(y)
        df = []
        for regressor in regressors:
            dist = [get_distance(x_[:-3],regressor.centre) for x_ in x]
            dist = np.array(dist).reshape((-1))
            df.append(
                regressor.predict(x).reshape((-1))/(dist+1)
            )

        
        for i in range(len(regressors)):
            columns.append("Local %s"%(i))
        columns.append("True")
        df = np.asarray(df)
        df = pd.DataFrame(np.concatenate((abo3.reshape((-1,len(columns)-1-len(regressors))),df.T,y.reshape((-1,1))),axis=1),columns=columns)
        df.to_excel("%s.local.xlsx"%outfiles[:-5],index=False)
        print("saved to ","%s.local.xlsx"%outfiles[:-5])   