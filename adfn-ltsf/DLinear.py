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

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.apprx = configs.apprx
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.apprx_target = configs.apprx_target
        self.channels = configs.enc_in
        self.apprx_flist = FuncPool()
        
        self.training_datas = []
        self.training_outputs = []
        self.save_switch = 0
        self.n_func = configs.n_func
        self.nan_counts = 0
        self.data_counts =0
        self.alter_all = configs.alter_all
        # add 1x1 scaling factor
        
        self.scaler = nn.Parameter(torch.ones(1,1), requires_grad=True)
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        elif self.apprx:
            
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.nonlinear = nn.LeakyReLU()
            self.apprx_seasonal_pre = nn.ModuleList()

            for i in range(self.n_func-1):
                self.apprx_seasonal_pre.append(MixedFunc(self.apprx_flist,config=configs))
                out_lens = self.apprx_seasonal_pre[i].forward_test(torch.randn(1, self.seq_len))
                self.apprx_seasonal_pre[i].initialize_betas_out(out_lens, self.seq_len)
                self.apprx_seasonal_pre[i].layer_idx = i

            

            self.apprx_seasonal = MixedFunc(self.apprx_flist,config=configs)
            self.apprx_trend = MixedFunc(self.apprx_flist,config=configs)
            self.alpha_mult = configs.alpha_mult
            self.alphas = self.initialize_alphas()
            self.aux_alphas = self.initialize_alphas()
            print("Initial_Alphas: ", self.alphas)
            
            with torch.no_grad():
                dummy_input = torch.randn(1, self.seq_len)
                out_length = self.apprx_seasonal.forward_test(dummy_input)
                out_trend_length = self.apprx_trend.forward_test(dummy_input)
                self.apprx_trend.initialize_betas_out(out_trend_length, self.pred_len)
                self.apprx_seasonal.initialize_betas_out(out_length, self.pred_len)
                self.apprx_trend.layer_idx = self.n_func-1
                self.apprx_seasonal.layer_idx = self.n_func-1
        

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
    def mid_out_forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        if self.apprx_target == 'seasonal':
            return seasonal_init, seasonal_output
        else:
            return trend_init, trend_output
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

            if self.apprx and self.save_switch == 1 and 'seasonal' in self.apprx_target:
                self.training_datas.append(seasonal_init.cpu().detach().numpy())
                self.training_outputs.append(seasonal_output.cpu().detach().numpy())
                
                # # save the training data and output for the seasonal component, in traffic_temps/
                # if self.data_counts % 100 == 0:
                #     np.save(f'traffic_temps/seasonal_data_{self.data_counts}.npy', np.array(self.training_datas))
                #     np.save(f'traffic_temps/seasonal_output_{self.data_counts}.npy', np.array(self.training_outputs))
                #     self.training_datas = []
                #     self.training_outputs = []
                # if self.data_counts == 938:
                #     np.save(f'traffic_temps/seasonal_data_{self.data_counts}.npy', np.array(self.training_datas))
                #     np.save(f'traffic_temps/seasonal_output_{self.data_counts}.npy', np.array(self.training_outputs))
                #     self.training_datas = []
                #     self.training_outputs = []
                # if self.data_counts == 989:
                #     np.save(f'traffic_temps/seasonal_data_{self.data_counts}.npy', np.array(self.training_datas))
                #     np.save(f'traffic_temps/seasonal_output_{self.data_counts}.npy', np.array(self.training_outputs))
                #     self.training_datas = []
                #     self.training_outputs = []

                self.data_counts += 1


            elif self.apprx and self.save_switch == 1 and 'trend' in self.apprx_target:
                self.training_datas.append(trend_init.cpu().detach().numpy())
                self.training_outputs.append(trend_output.cpu().detach().numpy())


        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
    def apprx_forward(self, apprx_data, gate):
        if 'seasonal' in self.apprx_target or gate == 0:
            for i in range(self.n_func-1):
                apprx_data = self.apprx_seasonal_pre[i].forward(apprx_data, self.alphas)
            apprx_output = self.apprx_seasonal.forward(apprx_data, self.alphas)
            # apprx_output = self.nonlinear(apprx_output)
        elif 'trend' in self.apprx_target or gate == 1:
            apprx_output = self.apprx_trend.forward(apprx_data, self.aux_alphas)
            apprx_output = apprx_output * self.scaler
        return apprx_output
    
    def alternative_forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if not self.alter_all:
            if 'seasonal' in self.apprx_target:
                for i in range(self.n_func-1):
                    seasonal_init = self.apprx_seasonal_pre[i].forward(seasonal_init, self.alphas)
                seasonal_output = self.apprx_forward(seasonal_init, 0)
                trend_output = self.Linear_Trend(trend_init)
            else:
                trend_output = self.apprx_forward(trend_init, 1)
                seasonal_output = self.Linear_Seasonal(seasonal_init)
        else:
            seasonal_output = self.apprx_forward(seasonal_init, 0)
            trend_output = self.apprx_forward(trend_init, 1)
            
        x = seasonal_output + trend_output
        return x.permute(0,2,1)

    def general_forward_test(self, x):
        len_res = []
        for i in range(len(self.layers)):
            len_out = self.layers[i].forward_test(x)
            len_res.append(len_out)
        return len_res

    def initialize_alphas(self):
        alpha = []
        '''mimic mixedop'''
        
        for i in range(self.n_func):
            # make enough alphas for sequence length
            # alpha.append(nn.Parameter(torch.ones(1, seq_len)/seq_len))
            alpha_val = torch.ones(1, len(self.apprx_seasonal.flist)*self.alpha_mult)/len(self.apprx_seasonal.flist)*self.alpha_mult
            alpha_noise = torch.randn(1, len(self.apprx_seasonal.flist)*self.alpha_mult)
            gamma = 1e-5
            alpha_noise = alpha_noise * gamma
            alpha.append(nn.Parameter(alpha_val + alpha_noise))
            # alpha.append(torch.ones(1, len(self.apprx_seasonal.flist)*self.alpha_mult)
            #              /len(self.apprx_seasonal.flist)*self.alpha_mult)
            # alpha.append(torch.randn(1, len(self.layers[0].flist)))
        alpha = nn.ParameterList(alpha)
        return alpha
    
        
    
    
    