import torch 



def get_topk_funcs(alphas,topk):
    '''get top k functions'''
    topk_funcs = []
    for i in range(len(alphas)):
        topk_funcs.append(torch.topk(alphas[i], topk, dim=1)[1])
        
    for i in range(len(topk_funcs)):
        print("At layer ", i, " topk functions are ", topk_funcs[i])    
    
    
    return topk_funcs 

def remove_other_funcs(funcs, topk_funcs, args):
    '''remove functions that are not in topk'''
    new_flist = []
    for i in range(len(funcs)):
        if i in topk_funcs:
            new_flist.append(funcs[i])
    # to me :
    # this is a bit tricky, the idea is to remove the functions that are not in topk_funcs
    args.func_list = new_flist

    return new_flist

def get_top_func_list_model(model,top_idx_list):
    funclist = {}
    funclist = model.get_top_func_names_layer(top_idx_list[0])
    return funclist   