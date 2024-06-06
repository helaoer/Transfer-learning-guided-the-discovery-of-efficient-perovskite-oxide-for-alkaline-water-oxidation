import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import csv
import numpy as np
import pickle
from paras import tuned_parameters,no_cluster,cluster_centres
from data_loader import data_loader
from sklearn.ensemble import GradientBoostingRegressor
name = "GBCT"

# for single data
class single_regressor():
    def __init__(self,value,centre,scaler):
        self.value = scaler.inverse_transform(value.reshape((-1,1))).reshape((-1))[0]
        self.centre = centre
        # self.scaler = scaler
    def predict(self,data):
        return np.array([self.value for i in range(data.shape[0])])

# for hybrid data
class multi_regressor():
    def __init__(self,models,centre,scaler):
        self.models = models
        self.centre = centre
        self.scaler = scaler
    def predict(self,data):
        out = np.mean([model.predict(data) for model in self.models],axis=0)
        return self.scaler.inverse_transform(out.reshape(-1,1))

def search(k_clusters=3):
    _,_,scaler,_,_ = data_loader()
    perform_test = []
    perform_cv = []
    data_size = []
    # train local regressor for each cluster
    for i in range(no_cluster):
        x,y,_,_,_ = data_loader(selected_clusters=[i],refit=False,scaler=scaler)
        print("cluster %s has %s data"%(i,x.shape[0]))
        if x.shape[0] < k_clusters:
            clf_best = single_regressor(np.mean(y.reshape((-1)),axis=0),cluster_centres[i],scaler)
            model_path = './models/clf_best_cluster_%s.pkl'%(i)
            with open(model_path, 'wb') as f:
                pickle.dump(clf_best, f)
                print("save to %s"%(model_path))
            # RMSE = 0
            perform_test.append(0)
            perform_cv.append(0)   
            data_size.append(x.shape[0])
        else:
            cv = KFold(n_splits=k_clusters, shuffle=True, random_state=2)
            clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, verbose=1, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=10)
            clf.fit(x, y.reshape(-1))
            print("{} {}-fold cross-validation neg_root_mean_squared_error: {:.3f}".format(name, x.shape[0], clf.best_score_))

            clf_ensemble = multi_regressor([clf.best_estimator_],cluster_centres[i],scaler)
            model_path = './models/clf_best_cluster_%s.pkl'%(i)
            with open(model_path, 'wb') as f:
                pickle.dump(clf_ensemble, f)
                print("save to %s"%(model_path))

            perform_cv.append(mean_squared_error(scaler.inverse_transform(y.reshape(-1,1)),clf_ensemble.predict(x).reshape(-1,1),squared=False)) 
            data_size.append(x.shape[0])

    perform_cv = np.asarray(perform_cv).reshape((-1,1))
    data_size = np.asarray(data_size).reshape((-1,1))

    data = pd.DataFrame(np.concatenate((data_size,perform_cv,),axis=1),columns=["no_of_data","cv_rmse",])
    data.to_excel("local_rmse_results.xlsx")

if __name__ == "__main__":     
    search()





