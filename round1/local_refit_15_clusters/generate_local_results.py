import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import pickle
from paras import no_cluster,filepath
from train import single_regressor,multi_regressor
class predictor():
    def __init__(self,):
        self.centres = []
        self.regressors = []
        for i in range(no_cluster):
            with open('./models/clf_best_cluster_%s.pkl'%(i),"rb") as f:
                reg = pickle.load(f)
                self.centres.append(reg.centre)
                self.regressors.append(reg)
                print("load ./models/clf_best_cluster_%s.pkl"%(i))
    def get_distance(self,data):
        dist = [np.linalg.norm(data[:4] - centre, axis=0) for centre in self.centres]
        return np.asarray(dist)

    def predict(self,data):
        distance = self.get_distance(data)
        # print(distance)
        result = np.asarray([reg.predict(data.reshape((1,-1))).item() for reg in self.regressors])
        # print(result)
        return  result,distance,result/(distance+1)

pred = predictor()
# print(pred.predict(np.asarray([1,1,1,1,1,1,1])))


def get_model_results(files, outfile_name="output"):
    for i,file in enumerate(files):
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                data = pd.read_excel(f)
                data = data.values
                x = data[:,3:-2]
                x = np.asarray(x, dtype=np.float64)
                y_pred = []
                for x_ in x:
                    _,_,y = pred.predict(x_)
                    y_pred.append(y)
                y_pred = np.asarray(y_pred)
                columns = ["ABO3", "v_sum", "valences", "x_0", "x_1", "x_2", "x_3","loading", "electrolyte", "substrate","y_true"]
                for dim in range(len(pred.centres)):
                    columns.append("y_%s"%dim)
                df = pd.DataFrame(np.concatenate((data[:,:-1],y_pred),axis=1), columns=columns)
                df.to_excel(f"{outfile_name}.local.slice {i + 1}.xlsx", index=False)
                print(f"Saved data to {outfile_name}.local.slice {i + 1}.xlsx")

import sys
sys.path.append('..') 
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

get_model_results([filepath],filepath)
