import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from functions import FuncPool
from mixedfunc import MixedFunc

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class DFNModel(nn.Module):
    """
    Decomposition-DFN
    """
    def __init__(self, n_func, in_length=None,pred_len=None):
        super(DFNModel, self).__init__()  
        self.layers = nn.ModuleList()
        self.flist_inst = FuncPool()
        self.n_func = n_func
        self.seq_len = in_length
        self.pred_len = pred_len


        # decomposition kernel size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.seasonal_layers = nn.ModuleList()
        self.trend_layers = nn.ModuleList()
        
        # for i in range(self.n_func):
        #     self.seasonal_layers.append(MixedFunc(self.flist_inst))
        #     self.trend_layers.append(MixedFunc(self.flist_inst))
        self.mixedfunc_seasonal = MixedFunc(self.flist_inst)
        self.mixedFunc_trend = MixedFunc(self.flist_inst)

        self.layers.append(self.mixedfunc_seasonal)
        self.layers.append(self.mixedFunc_trend)

        self.alphas = self.initialize_alphas()
        self.alphas_2 = self.initialize_alphas()

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.mid_outputs = []

        

        # for i in range(self.n_func):
        #     self.layers.append(MixedFunc(self.flist_inst))

        # self.softmax = nn.Softmax(dim=-1)
        # self.alphas = self.initialize_alphas()
        # self.activation = nn.LeakyReLU(negative_slope=0.2)
        # self.mid_outputs = []

        # if pred_len is None:
        #     self.lin_proj = nn.Identity()
        # else:
        #     self.lin_proj = nn.Linear(in_length, pred_len)

    def general_forward_test(self, x):
        len_res = []
        for i in range(len(self.layers)):
            len_out = self.layers[i].forward_test(x)
            len_res.append(len_out)
        return len_res
    
            
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) 
        
        seasonal_out = self.mixedfunc_seasonal.forward(seasonal_init, self.alphas)
        trend_out = self.mixedFunc_trend.forward(trend_init, self.alphas_2)

        out = seasonal_out + trend_out

        return out.permute(0,2,1)

    # def initialize_betas(self, idx, out_length, max_length):
    #     self.layers[idx].initialize_betas_in(out_length, max_length)
    
    # def softy_them_all(self):
    #     '''softmax all alphas'''
    #     for i in range(len(self.alphas)):
    #         self.alphas[i] = self.temperature_softmax(self.alphas[i],0.5)

    def update_funcs(self):
        '''update function list'''
        for i in range(len(self.layers)):
            self.layers[i].update_argnum()
        
    def initialize_alphas(self):
        alpha = []
        '''mimic mixedop'''
        for i in range(self.n_func):
            # make enough alphas for sequence length
            # alpha.append(nn.Parameter(torch.ones(1, seq_len)/seq_len))
            alpha.append(torch.ones(1, len(self.layers[0].flist))/len(self.layers[0].flist))
            # alpha.append(torch.randn(1, len(self.layers[0].flist)))
        
        alpha = nn.ParameterList(alpha)
        return alpha
    
    def temperature_softmax(self, alpha, temperature=1):
        '''softmax with temperature'''
        return F.softmax(alpha/temperature, dim=-1)

