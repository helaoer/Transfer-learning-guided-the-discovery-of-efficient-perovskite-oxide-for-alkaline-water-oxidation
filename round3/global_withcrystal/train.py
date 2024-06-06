import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import csv
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
name = "GBCT"
tuned_parameters = [
    {
        'n_estimators': [3, 5, 10, 20, 50],
        'learning_rate': [0.01, 0.1, 0.5],
        'subsample': [0.5, 0.8, 1.0],
        'max_depth': list(range(2, 10)),
        'min_samples_split': range(2,11,2),
        'min_samples_leaf': range(1,10,2),
        'max_features': [3,5,7],
        'random_state': [1],
    }
]

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

class cv_regressor():
    def __init__(self,models,scaler):
        self.models = models
        self.scaler = scaler
    def predict(self,data):
        out = np.mean([model.predict(data) for model in self.models],axis=0)
        return self.scaler.inverse_transform(out.reshape(-1,1))

def search(filepath):
    with open(filepath, "rb") as f:
        data = pd.read_excel(f)
        data = data.values
        x = data[:,3:-1]
        y = data[:,-1].reshape((-1,1))

    scaler_ = StandardScaler()
    scaler_.fit(x)
    x = scaler_.transform(x)

    scaler = StandardScaler()
    scaler.fit(y)
    y = scaler.transform(y)
    y = y.reshape((-1,1))

    perform_test = []
    perform_cv = []
    data_size = []

    # left_one_out
    cv = KFold(n_splits=x.shape[0], shuffle=True, random_state=2)
    clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, verbose=1, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=10)
    clf.fit(x, y.reshape(-1))
    print("{} {}-fold cross-validation neg_root_mean_squared_error: {:.3f}".format(name, x.shape[0], clf.best_score_))
    models = []
    test_true = []
    test_pred = []
    for train_index,test_index in cv.split(x,y):
        clf_best = GradientBoostingRegressor(**clf.best_params_)
        clf_best.fit(x[train_index],y[train_index].reshape((-1)))
        models.append(clf_best)
        test_pred += clf_best.predict(x[test_index]).reshape(-1).tolist()
        test_true += y[test_index].reshape(-1).tolist()

    clf_ensemble = cv_regressor(models,scaler)
    model_path = filepath[:-5]+".pkg"
    with open(model_path, 'wb') as f:
        pickle.dump(clf_ensemble, f)
        print("save to %s"%(model_path))

    test_true = np.asarray(test_true).reshape((-1,1))
    test_true = scaler.inverse_transform(test_true).reshape((-1,1))
    test_pred = np.asarray(test_pred).reshape((-1,1))
    test_pred = scaler.inverse_transform(test_pred).reshape((-1,1))

    perform_test.append(mean_squared_error(test_true,test_pred,squared=False))
    perform_cv.append(mean_squared_error(scaler.inverse_transform(y.reshape(-1,1)),clf_ensemble.predict(x).reshape(-1,1),squared=False)) 
    data_size.append(x.shape[0])
    perform_test = np.asarray(perform_test).reshape((-1,1))
    perform_cv = np.asarray(perform_cv).reshape((-1,1))
    data_size = np.asarray(data_size).reshape((-1,1))

    data = pd.DataFrame(np.concatenate((data_size,perform_test,perform_cv,),axis=1),columns=["no_of_data","test_rmse","cv_rmse",])
    data.to_excel(filepath[:-5]+".results.xlsx")