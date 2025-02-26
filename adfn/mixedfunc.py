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
    
    def __init__(self,  flistinst):
        super(MixedFunc, self).__init__()
        
        #based on exec func, lets make a instruction list for each function
        self.Flist = flistinst
        # number of arguments for each function
        self.num_args = []
        self.flist = [dir(self.Flist)[i] for i in range(len(dir(self.Flist))) if not dir(self.Flist)[i].startswith('_')]
        self.beta = nn.ParameterList()
        self.res_tensor = torch.Tensor([]).cuda()
        self.empty_tensor = torch.Tensor([]).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.residual = torch.Tensor([]).cuda()
        self.beta_i = nn.ParameterList()
        self.rounds = 0
        #get number of arguments 
        for func_name in self.flist:
            func = getattr(self.Flist, func_name)
            total_arg_num = func.__code__.co_argcount
            default_arg_num = 0 if func.__defaults__ is None else len(func.__defaults__)
            self.num_args.append(total_arg_num - default_arg_num - 1) # -1 for self
            

        
        
    def forward(self, x, alpha, debug_flag, weights=None):
        args = x
        weights = self.beta
        # weights_in = self.beta_i
        result = []
        # softmax beta
        idx = 0
        if debug_flag:
            # breakpoint()
            pass
        
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
            func_res = func(*args)

            self.check_nan(func_name,func_res)

            # func_res = self.sedate_func(func_res)
            func_res = alpha[0][i] * func_res
            result.append(func_res)
            
            args = x
            # print("func_name : ", func_name)
            # print("func_res : ", func_res)
     
        converted_result = self._convert_input(weights,result)
    

        result = torch.stack(converted_result,dim=0)
        # result = torch.stack(result,dim=0)
        if not debug_flag:
            result = torch.div(result,torch.max(torch.abs(result)))
        # result = torch.div(result,torch.max(torch.abs(result)))
        # else:
        #     print("result : ", result)
        #     print("max : ", )
        result = torch.sum(result,dim=0)

        return result

    def get_top_func_names_layer(self,topidx):
        idx_list = [x.item() for x in topidx[0]]
        top_lists = [self.flist[i] for i in idx_list]
        print("top functions : ", top_lists)
        
        return top_lists
        
        
        
        
    def beta_cleanup(self):
        '''clean up beta'''
        self.beta = nn.ParameterList()
        self.beta_i = nn.ParameterList()
    
    def sedate_func(self, x):
        # get x, normalize x and return
        return torch.div(x,torch.max(torch.abs(x)))

    def check_nan(self,func, x):
        if torch.isnan(x).any():
            print("nan output : ", x)
            print("nan check", torch.isnan(x).any())
            print("nan was at fuction", func) 
            print("Function : ", func)
            raise ValueError("nan detected")            
            
    def _convert_input(self,weights,input):
        ''' converts input to a list of vectors'''
        converted_input = []
        for i,res in enumerate(input):
            converted_input.append(self.n_to_k_conversion(weights[i], res))
        self.res_tensor = self.empty_tensor
        return converted_input


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
                # leave original value, don't change it
                arg = args[:arg_num]
                result.append(func(*arg))
                args = dummy_input
        
        # for i, res in enumerate(result):
        #     print(self.flist[i]," : ",res)
        len_result = [x.shape[-1] for x in result]
        # len_result = [len(x[0][0]) for x in result]
      
        return len_result
    
    
    def initialize_betas_in(self, out_length, max_length):
        '''initialize beta for Mixedfunc'''
        beta = nn.ParameterList()
        # initialize as rand
        # for i in range(len(out_length)):
        #     beta.append(torch.randn(max_length,out_length[i], requires_grad=True))
        # initialize as ones
        for i in range(len(out_length)):
            for ind_arg in range(self.num_args[i]):
                beta.append(torch.div(torch.ones(max_length,max_length, requires_grad=True),max_length))
                # beta.append(torch.randn(max_length,max_length, requires_grad=True))
                
        
        self.beta_i = beta



    def initialize_betas_out(self, out_length, max_length):
        '''initialize beta for Mixedfunc'''
        beta = nn.ParameterList()
        
        # initialize as rand
        # for i in range(len(out_length)):
        #     beta.append(torch.randn(max_length,out_length[i], requires_grad=True))
        # initialize as ones
        gamma = 1e-5
        for i in range(len(out_length)):
            values = torch.div(torch.ones(max_length,out_length[i], requires_grad=True),max_length)
            noise = torch.randn(max_length,out_length[i], requires_grad=True)
            values = values + noise * gamma
            beta.append(values)
            # beta.append(torch.div(torch.ones(max_length,out_length[i], requires_grad=True),max_length))
            # beta.append(torch.randn(max_length,out_length[i], requires_grad=True))
            
        self.beta = beta
        # print("initial beta : ", self.beta)
        


    def softy_them_all(self):
        '''softmax all betas'''
        for i in range(len(self.beta)):
            self.beta[i] = self.softmax(self.beta[i])
        

# if we execute this script

if __name__ == "__main__":

    print("Invoke train.py")
    