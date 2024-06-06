import torch.nn as nn
import torch
class AE(nn.Module):

    def __init__(self,args=[16,8,8,8,4]):
        super(AE, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(args[0], args[1]).cuda(),
            nn.LayerNorm(args[1]).cuda(),
            nn.ReLU().cuda(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(args[0]+args[1], args[2]).cuda(),
            nn.LayerNorm(args[2]).cuda(),
            nn.ReLU().cuda(),
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(args[0]+args[2], args[3]).cuda(),
            nn.LayerNorm(args[3]).cuda(),
            nn.ReLU().cuda(),
        )

        self.encoder4 = nn.Sequential(
            nn.Linear(args[0]+args[3], args[4]).cuda(),
        )       
        #####################################
        self.decoder1 = nn.Sequential(
            nn.Linear(args[4], args[3]).cuda(),
            nn.LayerNorm(args[3]).cuda(),
            nn.ReLU().cuda(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(args[4]+args[3], args[2]).cuda(),
            nn.LayerNorm(args[2]).cuda(),
            nn.ReLU().cuda(),
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(args[4]+args[2], args[1]).cuda(),
            nn.LayerNorm(args[1]).cuda(),
            nn.ReLU().cuda(),
        )

        self.decoder4 = nn.Sequential(
            nn.Linear(args[4]+args[1], args[0]).cuda(),
        )


    def forward(self, x):
        x_ = self.encoder1(x)
        x_ = torch.cat((x,x_),1)
        x_ = self.encoder2(x_)
        x_ = torch.cat((x,x_),1)
        x_ = self.encoder3(x_)
        x_ = torch.cat((x,x_),1)
        x_ = self.encoder4(x_)
        
        # x_out
        x_o = self.decoder1(x_)
        x_o = torch.cat((x_,x_o),1)
        x_o = self.decoder2(x_o)
        x_o = torch.cat((x_,x_o),1)
        x_o = self.decoder3(x_o)
        x_o = torch.cat((x_,x_o),1)
        x_o = self.decoder4(x_o)

        return x_o
    
    def encode(self, x):
        x_ = self.encoder1(x)
        x_ = torch.cat((x,x_),1)
        x_ = self.encoder2(x_)
        x_ = torch.cat((x,x_),1)
        x_ = self.encoder3(x_)
        x_ = torch.cat((x,x_),1)
        x_ = self.encoder4(x_)
        return x_

    def decode(self, x_):
        x_o = self.decoder1(x_)
        x_o = torch.cat((x_,x_o),1)
        x_o = self.decoder2(x_o)
        x_o = torch.cat((x_,x_o),1)
        x_o = self.decoder3(x_o)
        x_o = torch.cat((x_,x_o),1)
        x_o = self.decoder4(x_o)

        return x_o
