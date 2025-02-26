import torch
from datafeed import ClassificationDataset
import torch.nn as nn
import torch.optim as optim
import math 
from model import FuncControlModel
import copy
# Define your model
import random 
# data is sinx , x is 0 to 1
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm 
from torchmetrics import MeanAbsolutePercentageError
import datetime
from functions import FUNC_CLASSES
from func_manage import transfer_methods
from functions import FuncPool

EPOCH = 100
SEED = 7992







def train(args):
    
    if len(FUNC_CLASSES) != 0:
        class_keys = list(FUNC_CLASSES.keys())
        classes = [FUNC_CLASSES[key] for key in class_keys]
        
        if args.func_transfer == True:
            
            # get initial func pool method list
            flist = [dir(FuncPool)[i] for i in range(len(dir(FuncPool))) if not dir(FuncPool)[i].startswith('_')]
            print("Initial len: ", len(flist))
            print("Initial flist: ", flist)
            
            # transfer methods from classes to FuncPool
            transfer_methods(classes, FuncPool)
            
            # get final func pool method list
            flist = [dir(FuncPool)[i] for i in range(len(dir(FuncPool))) if not dir(FuncPool)[i].startswith('_')]
            print("Transfer_res : ", len(flist))
            print("Transfered flist: ", flist)
    
    else:
        print("No function classes transferred")
        
        
    now = datetime.datetime.now()                   
    # use only time and date                         
    now = now.strftime("%m-%d_%H:%M")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False                     
    torch.backends.cudnn.deterministic = True

    LR = args.lr
    EPOCH = args.epoch
    TYPE = args.type 
    bs = args.batch_size
    # settype = 'mixed'
    
    dataset  = ClassificationDataset(1000, settype=TYPE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True)
    
    ex_len = len(next(iter(dataloader))[0])
    
    model = FuncControlModel(args.layer, in_length = ex_len,cls_num=dataset.cls_num)
    
    print("model loaded")
    print("number of args for each functions : ", model.layers[0].num_args)
    
    # get flist of model
    
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.rand_like(next(iter(dataloader))[0])
    print("dummy : ", dummy_input)
    out_length = model.forward_test(dummy_input)
    max_length = max(out_length)
    
    
    # get func length and arg length, initialize. 
    model.initialize_betas(out_length,max_length)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_track = []
    model.cuda()
    # train
    model.train()
    
    from tqdm import tqdm 
    pbar = tqdm(range(EPOCH))
    cat_param = nn.Parameter(torch.rand(5,1)).cuda()
    
    for epoch in pbar:
        inner_loss_epoch = []
        pbar_inner = tqdm(enumerate(dataloader),leave=False, total=len(dataloader), desc="Epoch: {}".format(epoch+1))
        for i, data in pbar_inner:
          
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            # Backward pass
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e+5)
            alph_copy_before = copy.deepcopy(model.alphas[0])
            
            optimizer.step()
            
            inner_loss_epoch.append(loss.item())
        loss_track.append(sum(inner_loss_epoch)/len(inner_loss_epoch))
        pbar.set_postfix({'loss': loss_track[-1],'beta': model.layers[0].beta[0][0][0], 'alpha': model.alphas[-1][0][0]})

        # plot and save loss track
        plt.plot(loss_track, label='loss_track')
        plt.savefig(f'./result_img/loss_track_{EPOCH}_{TYPE}_{now}.png')
        plt.clf()
        df = pd.DataFrame([x.cpu().detach().numpy() for x in alph_copy_before])
        
        df.to_csv(f'./result_csv/alpha_track_{EPOCH}_{TYPE}_{now}.csv', mode='a', header=False, index=False)
        
        plt.clf()
        
        

    
    #after training, get func name and corresponding alpha. 
    print("funcs were : ", model.layers[0].flist)
    
    
    # save model pth
    torch.save(model.state_dict(), f'./result_model/model_{args.type}_{now}.pth')
            
    
    return model, now

            
def evaluate(model, args, now):
    EPOCH = args.epoch
    TYPE = args.type
    bs = args.batch_size
    if False:
        dataset  = ClassificationDataset(100, settype=TYPE)
        x = dataset.x
        y = dataset.y
        x = x.cuda()
        y = y.cuda()
        
        
        for i, sample in enumerate(x):
            test_output = model(sample)
            test_output = torch.argmax(test_output)
            # if we use mappers for here, how can we evaluate?
            
    else:
        print("Evaluation not implemented yet")
        
def plot_alpha_change(initial_alpha, final_alpha):
    # plot alpha change with bar graph
    plt.clf()
    plt.legend()
    plt.savefig('alpha_change.png')

def funcarg_exploration(num_args_list,func_list):
    # explore function and its arguments
    import inspect
    import pandas as pd 
    # using getattr
    # for each function, get the number of arguments
    func_arg_dict = {}
    for i, func in enumerate(func_list):
        args = inspect.getfullargspec(getattr(FuncPool, func)).args[1:]
        func_arg_dict[func] = [args]
    
    df = pd.DataFrame(func_arg_dict)
    df.to_csv('func_arg.csv', index=False)
    
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--type', type=str, default='sin')
    parser.add_argument('--fig_name', type=str, default='figname')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--func_transfer', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    model, now = train(args)
    evaluate(model,args, now)
    