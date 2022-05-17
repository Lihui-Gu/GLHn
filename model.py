import torch
import torch.nn.init as init
import numpy as np
# 引入torch.nn并指定别名
import torch.nn as nn
import torch.nn.functional as F

class GLHn(nn.Module):
    """
    input:  X: (375,360)    
         concept_matrix:(1, 2, 375, 375)
    output: (375)
    sample:
        >>X = torch.rand(375, 360)
        >>concept_matrix = torch.rand(1, 2, 375, 375)
        >>Y = GLHn(X, concept_matrix)
    """
    def __init__(self):
        super(GLHn, self).__init__()
        self.d_feat = 6
        self.hidden_size = 64
        self.conv1 = nn.Conv2d(1, 1, 1) 
        #两层GRU
        self.rnn = nn.GRU(
                input_size=self.d_feat,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )
        
        #线性层
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x, concept_matrix): 
        cm_out = self.conv1(concept_matrix)
        cm_out = F.relu(cm_out)
        x_1_out = torch.mm(cm_out.squeeze(), x)
        x_2_in = x_1_out.reshape(len(x_1_out), self.d_feat, -1) # [N, F, T]      
        x_2_in = x_2_in.permute(0, 2, 1) # [N, T, F]
        x_2_out, _ = self.rnn(x_2_in)
        x_2_out = x_2_out[:, -1, :]  
        # print(x_2_out.size())       #torch.Size([375, 64]) 375代表股票数量，64为每个股票的特征
        x_3_out = self.fc_1(x_2_out)
        x_3_out = self.leaky_relu(x_3_out)
        # print(x_3_out.size())
        x_4_in = x_2_out - x_3_out
        x_4_out = self.fc_1(x_4_in)
        x_4_out = self.leaky_relu(x_4_out)
        y_info = x_3_out + x_4_out
        pred = self.fc_out(y_info).squeeze()
        # print(pred.size())     
        return pred