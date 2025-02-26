import torch 
import torch.nn as nn
from mixedfunc import MixedFunc
from functions import FuncPool
import torch.nn.functional as F




class RefModel(nn.Module):
    '''Reference model, linear is baseline'''
    def __init__(self, num_inputs, num_outputs, num_layers, num_hidden):
        super(RefModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.mid_outputs = []
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, num_hidden))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_outputs))
        # self.one_that_conv = nn.Conv2d(1, 1, 28)


    def forward(self, x):
        self.mid_outputs = []
        for i in range(self.num_layers+1):
            x = self.layers[i](x)
            x = torch.relu(x)
            self.mid_outputs.append(x)
        return x

class RefConvModel(nn.Module):
    ''' Reference model, made with conv and linear projection'''
    def __init__(self, num_outputs):
        super(RefConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_outputs)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 9216)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class FuncControlModel(nn.Module):
    '''bring in mixedfunc and differentiable functions, initialize and manage weights'''
    def __init__(self, n_func, in_length=None,cls_num=None):
        super(FuncControlModel, self).__init__()
        self.layers = nn.ModuleList()
        self.flist_inst = FuncPool()
        self.n_func = n_func
        
        for i in range(self.n_func):
            self.layers.append(MixedFunc(self.flist_inst))
        
        self.softmax = nn.Softmax(dim=-1)
        
        # self.forward_test = self.layers[0].forward_test
        self.alphas = self.initialize_alphas()
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.mid_outputs = []
        
        if cls_num is None:
            self.lin_proj = nn.Identity()
        else:
            self.lin_proj = nn.Linear(in_length, cls_num)
            
    def general_forward_test(self, x):
        len_res = []
        for i in range(len(self.layers)):
            len_out = self.layers[i].forward_test(x)
            len_res.append(len_out)
        return len_res
        
    def forward(self, x):
        
        out_res = []
        self.mid_outputs = []
        # with torch.enable_grad():
        for i in range(self.n_func):
            if i == 0:
                out = x 
            out = self.layers[i](out, self.alphas[i])
            out = self.activation(out)
            self.mid_outputs.append(out)
            out_res.append(out)
            
            if i > 1: # residual connection. 
                out = torch.add(out, out_res[i-2])
        
            # print(out)
            if torch.isnan(out).any():
                print("nan output : ", out)
                print("nan check", torch.isnan(out).any())
                print("nan check", torch.isnan(self.alphas[i]).any())
                print("nan was at layer", i)
                exit()
    
        out = self.lin_proj(out)
        return out
    
    def initialize_betas(self, idx, out_length, max_length):
        self.layers[idx].initialize_betas_out(out_length, max_length)
        self.layers[idx].initialize_betas_in(out_length, max_length)
    
    def softy_them_all(self):
        '''softmax all alphas'''
        for i in range(len(self.alphas)):
            self.alphas[i] = self.temperature_softmax(self.alphas[i],0.5)

    def update_funcs(self):
        '''update function list'''
        for i in range(self.n_func):
            self.layers[i].update_argnum()
        
    def initialize_alphas(self):
        alpha = []
        '''mimic mixedop, check mixedop initialization of metanas'''
        for i in range(self.n_func):
            alpha.append(torch.ones(1, len(self.layers[0].flist))/len(self.layers[0].flist))
            # alpha.append(torch.randn(1, len(self.layers[0].flist)))
        
        alpha = nn.ParameterList(alpha)
        return alpha
    
    def temperature_softmax(self, alpha, temperature=1):
        '''softmax with temperature'''
        return F.softmax(alpha/temperature, dim=-1)


class AlternativeNeuralModel(nn.Module):
    '''Alternative model, linear is baseline'''
    def __init__(self, num_inputs=784, num_outputs=10, num_layers=3, num_hidden=784):
        super(AlternativeNeuralModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, num_hidden))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_outputs))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        return x
        
        
        
        
        