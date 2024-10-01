import matplotlib.pyplot as plt    
from data_loader import data_loader,DatasetConverter
from sklearn.model_selection import KFold,LeaveOneOut
import numpy as np
import pandas as pd
import os
import torch
from random import seed
from AE import AE
from sklearn.metrics import r2_score
from paras import args_list,n_splits,random_state,filepath,selected_args

def get_folds_result(args,n_splits=n_splits,random_state=random_state):
    torch.manual_seed(0)
    seed(0)
    x,_,scaler_x,_,_,_= data_loader(y_label=False)

    true_values_train = []
    pred_values_train = []
    true_values_test = []
    pred_values_test = []
    if n_splits == -1:
        skf = LeaveOneOut()
    else:
        skf = KFold(n_splits=n_splits,shuffle=True, random_state=random_state)
    for fold_no, (train_index, test_index) in enumerate(skf.split(x, x)):
        data_train = DatasetConverter(x[train_index],x[train_index],to_cuda=True)
        data_test = DatasetConverter(x[test_index],x[test_index],to_cuda=True)

        args_ = args.copy()
        args_.insert(0,data_train.x.size(dim=1))
        net = AE(args_)

        PATH = "encoder/encoder."
        for i in args:
            PATH = PATH + str(i) + "."
        PATH = PATH + str(fold_no) + ".net"

        if os.path.isfile(PATH):
            net.load_state_dict(torch.load(PATH))
            net.eval()
            # print("model loaded from {}".format(PATH))
        else:
            print("NO model loaded from {}".format(PATH))

        true_values_train.append(x[train_index])
        pred_values_train.append(net(data_train.x).cpu().detach().numpy())
        true_values_test.append(x[test_index])
        pred_values_test.append(net(data_test.x).cpu().detach().numpy())

    results = []
    # train
    title = ""
    for i in args:
        title += str(i) + "."
    r2_ = []
    plt.figure()
    for i in range(n_splits):
        true_vals = true_values_train[i]
        pred_vals = pred_values_train[i]
        r2 = r2_score(true_vals, pred_vals)
        r2_.append(r2)
        plt.scatter(true_vals, pred_vals, label="Folds %s r2 %.3f"%(i,r2))
    p1 = max(x[:,0])
    p2 = min(x[:,0])
    plt.plot([p1, p2], [p1, p2], 'b-', label="Equal")
    plt.xlabel('True Values (1st Dim)')
    plt.ylabel('Predicted Values (1st Dim)')
    plt.legend()
    r2_mean = np.mean(r2_)
    r2_std = np.std(r2_)
    plt.title("Train 5-Fold-CV R2: %.3f (%.3f)"%(r2_mean,r2_std))
    plt.savefig("result/%strain.png"%(title))
    plt.close()
    print("Train %s r2 mean %.3f std %.3f"%(args,r2_mean,r2_std))
    results.append(args)
    results.append(r2_mean)
    results.append(r2_std)

    # test
    title = ""
    for i in args:
        title += str(i) + "."
    r2_ = []
    plt.figure()
    for i in range(n_splits):
        true_vals = true_values_test[i]
        pred_vals = pred_values_test[i]
        r2 = r2_score(true_vals, pred_vals)
        r2_.append(r2)
        plt.scatter(true_vals, pred_vals, label="Folds %s r2 %.3f"%(i,r2))
    p1 = max(x[:,0])
    p2 = min(x[:,0])
    plt.plot([p1, p2], [p1, p2], 'b-', label="Equal")
    plt.xlabel('True Values (1st Dim)')
    plt.ylabel('Predicted Values (1st Dim)')
    plt.legend()
    r2_mean = np.mean(r2_)
    r2_std = np.std(r2_)
    plt.title("Test 5-Fold-CV R2: %.3f (%.3f)"%(r2_mean,r2_std))
    plt.savefig("result/%stest.png"%(title))
    plt.close()
    print("Test %s r2 mean %.3f std %.3f"%(args,r2_mean,r2_std))
    results.append(r2_mean)
    results.append(r2_std)
    return results

    
class predictor:
    def __init__(self,args=selected_args,n_splits=n_splits,filepath=filepath):

        torch.manual_seed(0)
        seed(0)
        x,_,scaler_x,_,_,_= data_loader(filepath=filepath,y_label=False)
        self.x = x
        self.x_dim = x.shape[1]
        self.scaler_x = scaler_x
        self.args = [self.x_dim]+args

        self.nets = []
        
        if n_splits == -1:
            n_splits = x.shape[0]

        for fold_no in range(n_splits):  
            net = AE(self.args)
            PATH = "encoder/encoder."
            for i in args:
                PATH = PATH + str(i) + "."
            PATH = PATH + str(fold_no) + ".net"

            if os.path.isfile(PATH):
                net.load_state_dict(torch.load(PATH))
                net.eval()
                self.nets.append(net)
                # print("model loaded from {}".format(PATH))
            else:
                print("NO model loaded from {}".format(PATH))
    
    # input = np.array
    def predict(self, input):
        y_pred = []
        for net in self.nets:
            y_pred.append(net(torch.from_numpy(input).type(torch.FloatTensor).cuda()).cpu().data.numpy())

        y_pred = np.asarray(y_pred)
        y_pred = np.mean(y_pred, axis=0)
        y_pred = np.reshape(y_pred, (-1,self.x_dim))

        y_pred = self.scaler_x.inverse_transform(y_pred)
        return y_pred

    def encode(self, input):
        y_pred = []
        for net in self.nets:
            y_pred.append(net.encode(torch.from_numpy(input).type(torch.FloatTensor).cuda()).cpu().data.numpy())

        y_pred = np.asarray(y_pred)
        y_pred = np.mean(y_pred, axis=0)
        y_pred = np.reshape(y_pred, (-1,self.args[-1]))
        return y_pred
    
    def get_result(self):
        title = ""
        for i in self.args[1:]:
            title += str(i) + "."

        plt.figure()
        y = self.scaler_x.inverse_transform(self.x)
        y_pred = self.predict(self.x)
        r2 = r2_score(y, y_pred)
        plt.scatter(y, y_pred, label="R2 %.3f"%(r2))
        p1 = max(max(y[:,0]),max(y_pred[:,0]))
        p2 = min(min(y[:,0]),min(y_pred[:,0]))
        plt.plot([p1, p2], [p1, p2], 'b-', label="Equal")
        plt.xlabel('True Overpotential (mV)')
        plt.ylabel('Predicted Overpotential (mV)')
        plt.legend()

        plt.title("5-Fold-CV")
        plt.savefig("result/%scv.png"%(title))
        plt.close()
        print("CV %s r2 %.3f"%(self.args[1:],r2))
        return [self.args[1:],r2]

if __name__ == "__main__":
    results = []
    for args_ in args_list:
        result = get_folds_result(args_)
        results.append(result)
    df = pd.DataFrame(results,columns=["args","train_r2_mean","train_r2_std","test_r2_mean","test_r2_std"]) 
    df.to_excel("results_r2_train_test.xlsx",index=False)   
    results = []
    for args_ in args_list:
        pred = predictor(args_)
        result = pred.get_result()
        results.append(result)
    df = pd.DataFrame(results,columns=["args","cv_r2"]) 
    df.to_excel("results_r2_cv.xlsx",index=False)      