import torch
import torch.nn as nn
from functions import FuncPool
import torch.optim as optim
import torch.nn.functional as F


class MixedFunc(nn.Module):
    '''Implementation of mixedfunc, going to follow mixedop.
    This is a function that takes in a list of functions and weights and returns a single function.
    Args:
        f_list: list of functions
        weights: list of weights
    '''
    
    def __init__(self,  flistinst, config=None):
        super(MixedFunc, self).__init__()
        self.config = config
        #based on exec func, lets make a instruction list for each function
        self.Flist = flistinst
        # number of arguments for each function
        self.num_args = []
        self.flist = [dir(self.Flist)[i] for i in range(len(dir(self.Flist))) if not dir(self.Flist)[i].startswith('_')]
        print("initial flist : ", self.flist)
        self.beta = nn.ParameterList()
        self.res_tensor = torch.Tensor([]).cuda()
        self.empty_tensor = torch.Tensor([]).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.residual = torch.Tensor([]).cuda()
        self.beta_i = nn.ParameterList()
        self.rounds = 0
        self.alpha_mult = config.alpha_mult
        self.soft_flag = config.soft_flag
        self.beta_alter_flag = config.beta_alter
        self.beta_regul = config.beta_regul
        self.layer_idx = 0
        self.epochs_now = 0
        
        

        #get number of arguments 
        for func_name in self.flist:
            func = getattr(self.Flist, func_name)
            total_arg_num = func.__code__.co_argcount
            default_arg_num = 0 if func.__defaults__ is None else len(func.__defaults__)
            self.num_args.append(total_arg_num - default_arg_num - 1) # -1 for self
            

        
        
    def forward(self, x, alpha, weights=None):
        
        args = x
        weights = self.beta
        
        result = []
        # softmax beta
        idx = 0
        # softmax alpha like in mixedop of darts
        if self.soft_flag:
            alpha = self.temp_alpha_softmax(alpha,1)

        for k in range(self.alpha_mult):
            for i in range(len(self.flist)):
                # change args to be the correct number of args
                corrected_args = []
                for j in range(self.num_args[i]):
                    corrected_args.append(args)
                args = corrected_args
                func_name = self.flist[i]
                func = getattr(self.Flist, func_name)
                # here, we need to apply beta to each argument. 
                # args = self.argumental_conversion(weights_in,args,idx)
                idx += self.num_args[i]
                try:
                    func_res = func(*args)
                    self.check_nan(func_name,func_res)
                except:
                    print("nan error occured")
                    exit()
                # func_res = self.sedate_func(func_res)

                if self.alpha_mult != 1:
                    index = i + k*len(self.flist)
                else:
                    index = i
                
                if self.config.n_func == 1:
                    func_res_k = alpha[0][k][index] * func_res
                else:
                    func_res_k = alpha[self.layer_idx][0][index] * func_res #remember that this is a scalar of first alpha. 
                # if layers are more than 1, we need to change the second index. 
                result.append(func_res_k)
                args = x

        if self.beta_alter_flag:# and (self.epochs_now+1) % 50 ==0:
           self.beta_alter(weights,mode='col')
        converted_result = self._convert_input(self.beta,result)
        # converted_result = result
        result = torch.stack(converted_result,dim=0)
        # result = torch.div(result,torch.max(torch.abs(result)))

        result = torch.sum(result,dim=0)
        return result

    def get_top_func_names_layer(self,topidx):
        idx_list = [x.item() for x in topidx[0]]
        multed_list = self.flist * self.alpha_mult
        top_lists = [multed_list[i] for i in idx_list]
        print("top functions : ", top_lists)
        
        return top_lists
        
        
        
        
    def beta_cleanup(self):
        '''clean up beta'''
        self.beta = nn.ParameterList()
        self.beta_i = nn.ParameterList()
    


    def check_nan(self,func, x):
        if torch.isnan(x).any():
            print("nan output : ", x)
            print("nan check", torch.isnan(x).any())
            print("nan was at fuction", func) 
            raise ValueError("nan detected")            
            
    def _convert_input(self,weights,input):
        ''' converts input to a list of vectors'''
        converted_input = []
        for i,res in enumerate(input):
            converted_input.append(self.n_to_k_conversion(weights[i%len(self.flist)], res))
        self.res_tensor = self.empty_tensor
        return converted_input

    def get_std(self,weights):
        '''get std of beta'''
        std_list = []
        for i in range(len(weights)):
            std_list.append(torch.mean(weights[i]))
        


        return torch.Tensor(std_list).cuda()

    def _get_max_elem_len(self,input):
        ''' returns the max length of certain element of a list, each element is a vector'''
        max_len = 0
        for elem in input:
            if len(elem) > max_len:
                max_len = len(elem)
        return max_len
        
    def n_to_k_conversion(self,weights,result):
        ''' converts n-length vector result to k-length vector'''
        res_tensor = self.res_tensor
        #got result of single function. 
        res_tensor = torch.nn.functional.linear(result,weights)
        return res_tensor     

    def beta_alter(self, weights, mode='row'):
        '''alter beta'''
        if mode == 'row':
            for i in range(len(weights)):
                beta_mean = torch.mean(weights[i], dim=0)
                self.beta[i] = beta_mean.repeat(weights[i].shape[0],1)

        elif mode == 'col':
            for i in range(len(weights)):
                beta_mean = torch.mean(weights[i], dim=1)
                beta_mean = beta_mean.unsqueeze(-1)
                self.beta[i] = beta_mean.repeat(1,weights[i].shape[1])
        
        elif mode == 'mono':
            for i in range(len(weights)):
                beta_mean = torch.mean(weights[i])
                self.beta[i] = beta_mean.repeat(weights[i].shape[0],weights[i].shape[1])
        
        elif mode == 'softmax':
            for i in range(len(weights)):
                self.beta[i] = self.softmax(weights[i])
        
        elif mode == 'rands':
            for i in range(len(weights)):
                self.beta[i] = torch.rand(weights[i].shape[0],weights[i].shape[1]).to("cuda")
        
        elif mode == 'naive':
            for i in range(len(weights)):
                self.beta[i] = torch.div(torch.ones(weights[i].shape[0],weights[i].shape[1]),weights[i].shape[0])
        
        

    def argumental_conversion(self,in_weights,arg,idx):
        arg_lists = []
        for i, single_arg in enumerate(arg):
            res_arg = torch.nn.functional.linear(single_arg,in_weights[idx+i])
            arg_lists.append(res_arg)
        return arg_lists

    def update_argnum(self):
        '''update argument number'''
        self.num_args = []
        for func_name in self.flist:
            func = getattr(self.Flist, func_name)
            total_arg_num = func.__code__.co_argcount
            default_arg_num = 0 if func.__defaults__ is None else len(func.__defaults__)
            self.num_args.append(total_arg_num - default_arg_num - 1)

    
    def forward_test(self,dummy_input):
        # run initially. 
        
        result = []
        args = dummy_input
        with torch.no_grad():

            for i in range(len(self.flist)):
                # change args to be the correct number of args
                # sample elems from initial tensor
                corrected_args = []
                for j in range(self.num_args[i]):
                    corrected_args.append(args)
                args = corrected_args
                func_name = self.flist[i]
                func = getattr(self.Flist, func_name)
                arg_num = self.num_args[i]
                arg = args[:arg_num]
                result.append(func(*arg))
                args = dummy_input
        len_result = [x.shape[-1] for x in result]
      
        return len_result
    
    
    def initialize_betas_in(self, out_length, max_length):
        '''initialize beta for Mixedfunc'''
        beta = nn.ParameterList()
        
        for i in range(len(out_length)):
            for ind_arg in range(self.num_args[i]):
                beta.append(torch.div(torch.ones(max_length,max_length, requires_grad=True),max_length))
                # beta.append(torch.randn(max_length,max_length, requires_grad=True))
                
        
        self.beta_i = beta



    def initialize_betas_out(self, out_length, max_length):
        '''initialize beta for Mixedfunc'''
        beta = nn.ParameterList()
        gamma = 0.001
        # initialize as ones
        for i in range(len(out_length)):
            beta_val = torch.div(torch.ones(max_length,out_length[i], requires_grad=True),max_length)
            beta_noise = torch.randn(max_length,out_length[i], requires_grad=True)
            beta_val = beta_val + gamma*beta_noise
            beta.append(beta_val)
        
        self.beta = beta.cuda()
        print("initial beta : ", self.beta)
        

    def softy_them_all(self):
        '''softmax all betas'''
        for i in range(len(self.beta)):
            self.beta[i] = self.softmax(self.beta[i])
        
    def temp_alpha_softmax(self,alpha,temp):
        '''softmax with temperature'''
        return self.softmax(alpha[0]/temp)
    
    def temp_beta_softmax(self,beta,temp):
        '''softmax with temperature'''
        return self.softmax(beta/temp)

# if we execute this script

if __name__ == "__main__":

    print("Invoke train.py")
    