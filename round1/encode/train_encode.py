import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from random import seed
from sklearn.preprocessing import StandardScaler
from data_loader import data_loader,DatasetConverter
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from paras import args_list,n_splits,random_state

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

def train(num_epochs=500,batch_size=1,args=[8,8,8,2],random_state=random_state,n_splits=n_splits):
    # for Reproducibility
    torch.manual_seed(0)
    seed(0)
    x,_,_,_,_,_= data_loader(y_label=False)

    if n_splits == -1:
        skf = LeaveOneOut()
    else:
        skf = KFold(n_splits=n_splits,shuffle=True, random_state=random_state)
    for fold_no, (train_index, test_index) in enumerate(skf.split(x, x)):

        data_train = DatasetConverter(x[train_index],x[train_index],to_cuda=True)
        data_test = DatasetConverter(x[test_index],x[test_index],to_cuda=True)

        dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False)

        PATH = "encoder/encoder."
        for i in args:
            PATH = PATH + str(i) + "."
        PATH = PATH + str(fold_no) + ".net"
        print(PATH)
        
        args_ = args.copy()
        args_.insert(0,data_train.x.size(dim=1))

        from AE import AE
        net = AE(args=args_).cuda()
        criterion = nn.MSELoss()
        from torch.optim import SGD,Adam
        optimizer = Adam(net.parameters(), lr=5e-4)
        best_net, best_loss = None, float('inf')

        loss = {}
        loss["train"] = []
        loss["test"] = []

        import numpy as np
        from time import time
        time_now = time()
        no_update = -1
        for epoch in range(num_epochs):
            no_update += 1
            if no_update < 100:
                loss_temp_train = []
                for data_train in dataloader_train:
                    x_train,y_train = data_train
                    optimizer.zero_grad()
                    y_pred = net(x_train)
                    loss_temp = criterion(y_pred,y_train)
                    loss_temp.backward()
                    optimizer.step()
                    loss_temp_train.append(loss_temp.cpu().detach().numpy())   
                loss_temp_train = np.mean(np.asarray(loss_temp_train))

                loss_temp_test = []
                for data_test in dataloader_test:
                    x_test,y_test = data_test
                    y_pred = net(x_test)
                    loss_temp = criterion(y_pred,y_test)
                    loss_temp_test.append(loss_temp.cpu().detach().numpy())
                loss_temp_test = np.mean(np.asarray(loss_temp_test))

                if loss_temp_test < best_loss:
                    best_loss = loss_temp_test
                    print("Best test_loss achieved in Epoch %5d/%5d:%6.4f train_loss: %6.4f" % (epoch+1, num_epochs, best_loss, loss_temp_train))
                    torch.save(net.state_dict(), PATH)
                    no_update = -1

                if (epoch % 10 == 9):
                    time_now = time() - time_now
                    print("Epoch %5d/%5d Time used: %6.4f" % (epoch+1, num_epochs, time_now))
                    time_now = time()

                loss["train"].append(loss_temp_train)
                loss["test"].append(loss_temp_test)
            else:
                continue

        torch.cuda.empty_cache()
        
        import matplotlib.pyplot as plt    
        plt.figure()
        import numpy as np
        x_axis = np.arange(len(loss["train"]))
        plt.plot(x_axis, loss["train"],label="Train loss",color='red')
        plt.plot(x_axis, loss["test"],label="Test loss",color='green')
        plt.ylim([0,1])
        plt.grid()
        plt.legend()
        plt.savefig(str(PATH+".png"))
        plt.close()

for args_ in args_list:
    train(num_epochs=1000,batch_size=1,args=args_)

