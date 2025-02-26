import torch
from datafeed import RegressionDataset
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
from func_prune import get_topk_funcs, get_top_func_list_model, remove_other_funcs

EPOCH = 100



def alpha_optim(alpha, lr):
    ''' deprecated func'''
    for row in alpha:
        row.grad.data = row.grad.data 
        row.data = lr * row.data.add(-row.grad.data)
        row.grad.data.zero_()
    
    return alpha



def train(args, model=None):
    print("Rounds : ", args.rounds)

    if args.rounds == 0:
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
    else:
        print("Func transfer with pruning")
        # got new flist, initialize alpha again.
        for i in range(len(model.layers)):
            print("At layer ", i, " funcs are : ", model.layers[i].flist)
       

        
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
    
    dataset  = RegressionDataset(10000, args=args,settype=TYPE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True)

    if args.rounds == 0:
        model = FuncControlModel(args.layer, in_length=None, cls_num=None)
    else: 
        # re-initialize alphas based on new flist
        model.alphas = model.initialize_alphas()
        model.update_funcs()

    
    print("model loaded")
    print("number of args for each functions : ", model.layers[0].num_args)
    
    # get flist of model
    if args.onehot == 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    dummy_input = torch.rand_like(next(iter(dataloader))[0])
    len_results = model.general_forward_test(dummy_input)
    
    
    # get func length and arg length, initialize. 
    for idx in range(len(len_results)):
        max_idx_len = max(len_results[idx])
        model.initialize_betas(idx,len_results[idx],max_idx_len)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_track = []
    model.cuda()
    # train
    model.train()
    answers = [0.0, 0.1, 0.2, 0.3, 0.4]
    from tqdm import tqdm 
    pbar = tqdm(range(EPOCH))
    cat_param = nn.Parameter(torch.rand(5,1)).cuda()
    for epoch in pbar:
        inner_loss_epoch = []
        inner_viz = []

        pbar_inner = tqdm(enumerate(dataloader),leave=False, total=len(dataloader), desc="Epoch: {}".format(epoch+1))
        correct = 0
        
        for i, data in pbar_inner:
           
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # one closest to the answer
        
            # outputs = torch.sum(outputs, dim=-1)
            if args.onehot == 1:
                outputs = torch.sum(outputs, dim=2)
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
            # Backward pass
            loss.backward()


            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e+5)
            alph_copy_before = copy.deepcopy(model.alphas[0])
            
            optimizer.step()
            
           

            with torch.no_grad():
                 # get correctness here
                # Calculate the difference between outputs and each answer
                differences = [torch.abs(outputs - answer) for answer in answers]
                # Stack the differences along a new dimension
                differences = torch.stack(differences, dim=-1)
                # Find the index of the minimum difference along the last dimension
                min_index = torch.argmin(differences, dim=-1)
                # Use this index to get the closest answer
                guess = [answers[i] for i in min_index]
                correct += torch.sum(torch.Tensor(guess).cuda() == targets)
            
            
            
            # pbar.set_postfix({'loss': loss.item()})
            
            inner_loss_epoch.append(loss.item())
            # get accuracy with correct

            # print("outputs argmax : ", outputs)
            # accuracy = (correct / (bs))/len(pbar_inner)
            pbar_inner.set_postfix({'loss': loss.item(), 'alpha': model.alphas[0][0].data[0].item()})

        pbar.set_postfix({'loss': loss.item(), 'accuracy': (correct.item() / len(dataset))/bs})
        loss_track.append(sum(inner_loss_epoch)/len(inner_loss_epoch))

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
    
def restore_grad_health(model):
    for n, p in model.named_parameters():
        if p.grad is not None:
            if torch.any(torch.isnan(p.grad)):
                p.grad = torch.zeros_like(p.grad)
        else:
            continue
            
            
def evaluate(model, args, now):
    EPOCH = args.epoch
    if args.type == "sin":
        test_input = torch.linspace(0, 2*np.pi, 100)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point, debug_flag=True)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.sin(test_input ).squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')
        
        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/sin_in_outs.csv', index=False)
        
        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Sin')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')

        
    elif args.type == 'sin_cos':

        test_input = torch.linspace(0, 2*np.pi, 1000)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.sin(test_input).squeeze(dim=-1) + torch.cos(test_input).squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/sin_cos_in_outs.csv', index=False)

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Sin+Cos')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')


    elif args.type == 'exp':
        test_input = torch.linspace(0, 2*np.pi, 100)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.exp(test_input).squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/exp_in_outs.csv', index=False)

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Exp')
        plt.legend() 
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')
    
    elif args.type == 'exp_sin_double':
        test_input = torch.linspace(0, 2*np.pi, 100) - np.pi
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.exp(test_input).squeeze(dim=-1) + torch.sin(test_input).squeeze(dim=-1) + 2 * test_input.squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/exp_sin_double_in_outs.csv', index=False)

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Exp+Sin+2x')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')
        
    elif args.type == 'exp_log':
        # x should be more than 1, less than pi
            test_input = torch.linspace(1, 2*np.pi, 100)
            test_input = torch.clamp(test_input, min=1, max=np.pi)
            test_input = torch.unsqueeze(test_input, dim=-1)
            test_input = test_input.cuda()
            test_res=[]
            for point in tqdm(test_input):
                test_output = model(point)
                test_res.append((point.item(), test_output.item()))
            
            gt = torch.exp(test_input).squeeze(dim=-1) + torch.log(test_input).squeeze(dim=-1)
            
            # print beta
            torch.save(model.state_dict(), 'model.pth')

            # save test_res to csv
            df = pd.DataFrame(test_res, columns=['input', 'output'])
            df.to_csv('result_out_csv/exp_log_in_outs.csv', index=False)

            # plt plotting test_output only
            plt.clf()
            # plot point as a x axis and test_output as y axis,
            plt.plot(*zip(*test_res), label='MixedFunc')
            plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Exp+Log')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')


    elif args.type == "exp_sin":
        test_input = torch.linspace(0, 2*np.pi, 100)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.exp(test_input).squeeze(dim=-1) + torch.sin(test_input).squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/exp_sin_in_outs.csv', index=False)

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Exp+Sin')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')
        
    elif args.type == 'log_sin':
        test_input = torch.linspace(1, np.pi, 100)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.log(test_input).squeeze(dim=-1) + torch.sin(test_input).squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/log_sin_in_outs.csv', index=False)

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Log+Sin')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')

    elif args.type == 'rexp_sin':
        test_input = torch.linspace(-np.pi, np.pi, 100)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point)
            test_res.append((point.item(), test_output.item()))

        gt = torch.exp(torch.reciprocal(test_input)).squeeze(dim=-1) + torch.sin(test_input).squeeze(dim=-1)

        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/rexp_sin_in_outs.csv', index=False)
        
        plt.clf()
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Exp+Sin')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_{args.layer}_{now}.png')





    elif args.type == 'cos':
        test_input = torch.linspace(0, 2*np.pi, 100)
        test_input = torch.unsqueeze(test_input, dim=-1)
        test_input = test_input.cuda()
        test_res=[]
        for point in tqdm(test_input):
            test_output = model(point, debug_flag=True)
            test_res.append((point.item(), test_output.item()))
        
        gt = torch.cos(test_input).squeeze(dim=-1)
        
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # save test_res to csv
        df = pd.DataFrame(test_res, columns=['input', 'output'])
        df.to_csv('result_out_csv/cos_in_outs.csv', index=False)

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.plot(*zip(*test_res), label='MixedFunc')
        plt.plot(test_input.cpu().detach().numpy(), gt.cpu().detach().numpy(), label='Cos')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.layer}_{now}.png')
    elif args.type == 'modularsum':
        if args.onehot == 0:
            test_input = torch.randint(low=0, high=100, size=(100, 2)).float()
            test_input = test_input.cuda()

            test_res=[]
            # #use only 100 samples
            test_input = test_input[:50]
            
            
            for i, point in enumerate(tqdm(test_input)):
                test_output = model(point)
                test_output = torch.sum(test_output)   
                test_res.append(( test_output.item()))
            
            gt = torch.div(torch.sum(test_input,dim=1) % 5,10).unsqueeze(dim=-1)
            gt = gt.cuda()
            gt = gt[:50]
        
            # print beta
            torch.save(model.state_dict(), f'model_{args.type}.pth')

            # plt plotting test_output only
            # plot each 10 points for better visualization
            for i in range(0, len(test_res), 10):
                plt.clf()
                # plot point as a x axis and test_output as y axis,
                plt.scatter(range(i,i+10), test_res[i:i+10], label='test_output', color='blue', s=2)
                # plot gt with tiny red dots
                plt.scatter(range(i,i+10), gt.cpu().detach().numpy()[i:i+10], label='gt', color='red', s=2)
                
                # for each gt, plot black vertical line for matching
            
                plt.legend()
                plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_layer_{args.layer}_{now}_{i}.png')
        
        elif args.onehot == 1:
            test_input = torch.randint(low=0, high=100, size=(100, 2)).float()
            test_input = test_input.cuda()
            test_input = torch.nn.functional.one_hot(test_input.long()).float()
            test_res=[]
            # #use only 100 samples
            test_input = test_input[:50]
            
            
            for i, point in enumerate(tqdm(test_input)):
                test_output = model(point)
                # test_output = torch.sum(test_output)   
                test_res.append(( test_output.item()))
            
            gt = torch.div(torch.sum(test_input,dim=1) % 5,10).unsqueeze(dim=-1)
            gt = gt.cuda()
            gt = gt[:50]
        
            # print beta
            torch.save(model.state_dict(), f'model_{args.type}.pth')

            # plt plotting test_output only
            # plot each 10 points for better visualization
            for i in range(0, len(test_res), 10):
                plt.clf()
                # plot point as a x axis and test_output as y axis,
                plt.scatter(range(i,i+10), test_res[i:i+10], label='test_output', color='blue', s=2)
                # plot gt with tiny red dots
                plt.scatter(range(i,i+10), gt.cpu().detach().numpy()[i:i+10], label='gt', color='red', s=2)
                
                # for each gt, plot black vertical line for matching
            
                plt.legend()
                plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_layer_{args.layer}_{now}_{i}.png')
            



    elif args.type == 'california':
        print("evaluating on california")
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        data, target = california.data, california.target
        # use only first hundreds
        data = data[:100]
        target = target[:100]
        
        test_input = torch.Tensor(data)
        test_input = test_input.cuda()
        test_res=[]
        for i, point in enumerate(tqdm(test_input)):
            test_output = model(point)
            test_res.append((i, test_output.item()))
        
        gt = torch.div(torch.Tensor(target), torch.max(torch.Tensor(target)))
        gt = gt.cuda()
        # print beta
        torch.save(model.state_dict(), 'model.pth')

        # plt plotting test_output only
        plt.clf()
        # plot point as a x axis and test_output as y axis,
        plt.scatter(*zip(*test_res), label='test_output')
        plt.scatter(range(i+1), gt.cpu().detach().numpy(), label='gt', color='red')
        plt.legend()
        plt.savefig(f'./result_output/test_output_{EPOCH}_{args.type}_{args.fig_name}_layer_{args.layer}_{now}.png')
        
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
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--type', type=str, default='exp')
    parser.add_argument('--fig_name', type=str, default='figname')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--func_transfer', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--progress', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--onehot', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=0)
    args = parser.parse_args()

    # topk_numbs = [7, 5]

    topk_numbs = [5]

    model = None
    for round in range(len(topk_numbs)):
        print("Round : ", round)
        model, now = train(args , model)
        top_func_idx_list = get_topk_funcs(model.alphas, topk_numbs[round])
        top_func_name_list = get_top_func_list_model(model, top_func_idx_list)

        # evaluate(model,args, now)
        # prune! 
        for i in range(args.layer):
            new_flist = remove_other_funcs(model.layers[i].flist, top_func_idx_list[i], args)
            model.layers[i].flist = new_flist
            print("new flist : ", new_flist)

        args.rounds += 1

    
    alpha_save = sorted(model.alphas[0][0].cpu().detach().numpy(),reverse=True)[:topk_numbs[round]]
    # model, now = train(args,model)    
    alpha_df = pd.DataFrame(alpha_save)
    alpha_df = alpha_df.T
    alpha_df.to_csv(f'./result_csv/alpha_{args.type}.csv', mode='a', header=False, index=False)
    # save flist to csv file in 'a' mode, single layer
    df = pd.DataFrame(top_func_name_list)
    # top_func_name_list = get_top_func_list_model(model, top_func_idx_list)
    # in same row
    df = df.T
    df.to_csv(f'./result_csv/flist_{args.type}.csv', mode='a', header=False, index=False)
    
    # # save alphas
    # # df = pd.DataFrame([x.cpu().detach().numpy() for x in model.alphas[0]])
    # # new df 7,27,1,8,10 only
    # # picked_alphas = [model.alphas[0][0][i].cpu().detach().numpy() for i in top_func_idx_list[0][0]]
    # alphas = sorted(model.alphas[0][0].cpu().detach().numpy(),reverse=True)
    # df = pd.DataFrame(alphas)
    # df.to_csv(f'./result_csv/alpha_{args.type}.csv', mode='a', header=False, index=False)

    # # save alphas and funcs
    # # evaluate(model,args, now)
    #     #final round
    # # print("found funcs : ", model.layers[0].flist)

    # # model, now = train(args, model)
    # evaluate(model,args, now)
    # #plot epoch accuracy
    
    # top_func_idx_list = get_topk_funcs(model.alphas,topk_numbs[-1])
    # for i in range(args.layer):
    #     top_func_name_list = get_top_func_list_model(model, top_func_idx_list)
    #     # print("top func name list : ", top_func_name_list)
    #     # sort list according to top_func_idx.
    #     # sorted_last_list = [x for _, x in sorted(zip(top_func_idx_list[i][0], model.layers[i].flist))]
    #     # print("sorted last list : ", sorted_last_list)
    #     # print("top func idx list : ", top_func_idx_list)

    #     print("alphas : ", [x.item() for x in model.alphas[i][0]], "funcs : ", model.layers[i].flist)
    #     print("Betas : ", model.layers[i].beta[0].data)
    #     # print("found alpha : ", [model.alphas[i][idx] for idx in top_func_idx_list])

    #     # print("found alpha :", [np.round(model.alphas[0].detach().cpu().numpy()[0][i], 4) for i in top_func_idx_list[0].cpu().numpy()])