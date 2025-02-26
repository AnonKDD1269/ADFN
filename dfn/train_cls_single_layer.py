import torch
from datafeed import RegressionDataset, ClassificationDataset
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
import torchvision 
import torchvision.transforms as transforms
from func_prune import get_topk_funcs, remove_other_funcs
#import pdb


EPOCH = 100
SEED = 7992




def alpha_optim(alpha, lr):
    ''' deprecated func'''
    for row in alpha:
        row.grad.data = row.grad.data 
        row.data = lr * row.data.add(-row.grad.data)
        row.grad.data.zero_()
    
    return alpha



def train(args, model=None, ref_model=None):
    if args.rounds == 0:
        print("Round 0, no pruning")
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
        print("functions have been pruned, begin pruning")
        # got new flist, initialize alpha again.
        for i in range(len(model.layers)):
            print("At layer ", i, " funcs are : ", model.layers[i].flist)
        

    # import code 
    # code.interact(local=locals())
        
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
    SIZE = 28
    # settype = 'mixed'
    # mnist

    dataset  = ClassificationDataset(settype=TYPE)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    # ex_len = len(next(iter(dataloader))[0][1])
    
    if args.rounds == 0:
        model = FuncControlModel(args.layer, in_length =dataset.in_length)
    else: 
        # re-initialize alphas based on new flist
        model.alphas = model.initialize_alphas()
        model.update_funcs()
        
    
    print("model loaded")
    print("number of args for each functions : ", model.layers[0].num_args)
    
    # get flist of model
    
    criterion = nn.MSELoss()
    # dataloader output type
    dummy_input = torch.rand_like(next(iter(dataloader))[0])
    # dummy_input = dummy_input.view(-1,SIZE**2)
    print("dummy : ", dummy_input)
    
    len_results = model.general_forward_test(dummy_input)
        
    # out_length = model.forward_test(dummy_input)
    # max_length = max(out_length)
    
    # get func length and arg length, initialize. 
    for idx in range(len(len_results)):
        max_idx_len = max(len_results[idx])
        model.initialize_betas(idx,len_results[idx],max_idx_len)
        
        
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_track = []
    model.cuda()
    # train
    model.train()
    
    from tqdm import tqdm 
    pbar = tqdm(range(EPOCH))
    lin_params = torch.ones(10,SIZE**2).cuda()
    epoch_acc = []
    
    if args.mode =='solo':
        for epoch in pbar:
            inner_loss_epoch = []
            inner_viz = []
            corr = 0
            
            pbar_inner = tqdm(enumerate(dataloader),leave=False, total=len(dataloader), desc="Epoch: {}".format(epoch+1))
            for i, data in pbar_inner:
                
                # if i == 32:
                #     breakpoint()
                    
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                with torch.no_grad():
                    #count correct predictions
                    _, predicted = torch.max(outputs, 1)
                    _, gt = torch.max(targets, 1)
                    corr += (predicted == gt).sum().item()
                    
                # Backward pass
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e+5)
                alph_copy_before = copy.deepcopy(model.alphas[0])
                
                optimizer.step()
                
                inner_loss_epoch.append(loss.item())
                pbar_inner.set_postfix({'loss': loss.item(),  'alpha': model.alphas[-1][0][0], 'acc': (corr/(i+1))/bs})
            loss_track.append(sum(inner_loss_epoch)/len(inner_loss_epoch))
            pbar.set_postfix({'loss': loss_track[-1], 'alpha': model.alphas[-1][0][0], 'acc': (corr/(i+1))/bs})

            # plot and save loss track
            plt.plot(loss_track, label='loss_track')
            plt.savefig(f'./result_img/loss_track_{EPOCH}_{TYPE}_{now}.png')
            plt.clf()
            df = pd.DataFrame([x.cpu().detach().numpy() for x in alph_copy_before])
            
            df.to_csv(f'./result_csv/alpha_track_{EPOCH}_{TYPE}_{now}.csv', mode='a', header=False, index=False)
            
            plt.clf()
            
            epoch_acc.append(corr/len(dataloader.dataset))

        
        #after training, get func name and corresponding alpha.
        for i in range(len(model.layers)):
            print("At layer ", i, " funcs were : ", model.layers[i].flist) 
        
        # save model pth
        torch.save(model.state_dict(), f'./result_model/model_{args.type}_{now}.pth')
        
    else:
        
        ref_model.cuda()
        mape = MeanAbsolutePercentageError().cuda()
        appr_criterion = nn.MSELoss()
        mix_param = 0
        # model.register_parameter('mix_param', mix_param)
        # mix_param.requires_grad = True

        for epoch in pbar:
            inner_loss_epoch = []
            inner_viz = []
            corr = 0
            
            pbar_inner = tqdm(enumerate(dataloader),leave=False, total=len(dataloader), desc="Epoch: {}".format(epoch+1))
            for i, data in pbar_inner:
                
                mid_loss = []
                # if i == 32:
                #     breakpoint()
                
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer.zero_grad()
                with torch.no_grad():
                    # ref forward
                    ref_output = ref_model(inputs)
                    # separate forward layer by layer
                    layer_out = ref_model.layers[0](inputs)
                    target_out = ref_model.layers[1](layer_out)
                # Forward pass
                outputs = model(layer_out)

                softed_outputs = torch.nn.functional.softmax(outputs, dim=1)
                softed_target_outputs = torch.nn.functional.softmax(target_out, dim=1)


         
                loss = criterion(softed_outputs,softed_target_outputs)

                # with torch.no_grad():
                #     _, predicted = torch.max(outputs, 1)
                #     _, gt = torch.max(targets, 1)
                #     corr += (predicted == gt).sum().item()

                # # get mid loss
                # for j in range(len(model.layers)):
                #     ref_mid_outs = ref_model.mid_outputs[j]
                #     mid_outs = model.mid_outputs[j]
                #     soft_ref_mid_outs = torch.nn.functional.softmax(ref_mid_outs, dim=1)
                #     soft_mid_outs = torch.nn.functional.softmax(mid_outs, dim=1)
                #     mid_loss.append(appr_criterion(soft_mid_outs, soft_ref_mid_outs))
                    
                # # mid_loss_sum = sum(mid_loss)
                # mid_loss_sum = torch.mean(torch.stack(mid_loss))
                
                # softmax it 
                # outputs_soft = torch.nn.functional.softmax(outputs, dim=1)
                # ref_output_soft = torch.nn.functional.softmax(ref_output, dim=1)
                # print("output shape : ", outputs_soft.shape)
                # print("ref shape : ", ref_output_soft.shape)
                # appr_loss = appr_criterion(outputs_soft, ref_output_soft) 
                total_loss = loss
                # total_loss = appr_loss
                # check acc 
           
                
                # # get MAPE for logging between model and ref model
                # with torch.no_grad(): 
                #     mape_val = mape(outputs, ref_output)
                
                # Backward pass
                total_loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e+5)
                alph_copy_before = copy.deepcopy(model.alphas[0])
                
                optimizer.step()
                
                inner_loss_epoch.append(loss.item())
                pbar_inner.set_postfix({'loss': total_loss.item(),  
                                        'alpha': model.alphas[-1][0][0],
                                        # 'mape': mape_val.item(),
                                        # 'appr_loss': mid_loss_sum.item(),
                                        'corr': corr , 'total': len(dataloader.dataset),'acc': (corr/(i+1))/bs})
                
            loss_track.append(sum(inner_loss_epoch)/len(inner_loss_epoch))
            pbar_inner.set_postfix({'loss': total_loss.item(),
                                    'alpha': model.alphas[-1][0][0],
                                    # 'mape': mape_val.item(),
                                    # 'appr_loss': mid_loss_sum.item(),
                                    'acc': (corr/(i+1))/bs})
            
            # plot and save loss track
            plt.plot(loss_track, label='loss_track')
            plt.savefig(f'./result_img/loss_track_{EPOCH}_{TYPE}_{now}_ref.png')
            plt.clf()
            df = pd.DataFrame([x.cpu().detach().numpy() for x in alph_copy_before])
            
            df.to_csv(f'./result_csv/alpha_track_{EPOCH}_{TYPE}_{now}_ref.csv', mode='a', header=False, index=False)
            
            plt.clf()
            
            epoch_acc.append(corr/len(dataloader.dataset))

        
        #after training, get func name and corresponding alpha.
        for i in range(len(model.layers)):
            print("At layer ", i, " funcs were : ", model.layers[i].flist) 
        
        # save model pth
        torch.save(model.state_dict(), f'./result_model/model_{args.type}_{now}_ref.pth')
            
    return model, now, epoch_acc
    

            
def evaluate(model, args, now):
    EPOCH = args.epoch
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize([28,28])])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
    
    model.eval()
    loss_track = []
    model.cuda()
    corr = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs = inputs.view(-1, 28**2)
        outputs = model(inputs)
        targets = torch.nn.functional.one_hot(targets, num_classes=10).float()
        corr += (torch.argmax(outputs, dim=1) == torch.argmax(targets, dim=1)).sum().item()
    
    print("Validation accuracy : ", corr/len(val_loader.dataset))
    
      
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

def get_top_func_list_model(model,top_idx_list):
    funclist = {}
    for i in range(len(model.layers)):
        funclist[i] = model.layers[i].get_top_func_names_layer(top_idx_list[i])
    return funclist   
    
def visualize_epoch_acc(epoch_acc_dict, now,args):
    # plot epoch accuracy
    for key in epoch_acc_dict.keys():
        plt.plot(epoch_acc_dict[key], label=f'Round {key}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.legend()
    plt.savefig(f'./result_img/epoch_acc_{args.type}_{now}_seed_{args.seed}_{args.batch_size}_{args.fig_name}.png')


def ref_model_training(args):
    # Set up the training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    epochs = 30
    batch_size = 128

    # Load the MNIST dataset
    if args.type == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    elif args.type == 'cifar10':
        # to grayscale
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.Grayscale(num_output_channels=1), transforms.Resize([28,28])])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the model architecture
    from model import RefModel
    import os

    model = RefModel(784, 10, 3, 784)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    pbar = tqdm(range(epochs), desc="Ref Training")
    if os.path.exists(f'./result_model/ref_model_{args.type}_single.pth'):
        model.load_state_dict(torch.load(f'./result_model/ref_model_{args.type}_single.pth'))
        print("Ref model loaded")

        return model
    
    for epoch in pbar:
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss/len(train_loader)})

    torch.save(model.state_dict(), f'./result_model/ref_model_{args.type}_single.pth')

    print("Ref Training complete!")
    return model 


import argparse 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--type', type=str, default='mnist')
    parser.add_argument('--fig_name', type=str, default='figname')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--func_transfer', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--len_funcs', type=int, default=0)
    parser.add_argument('--func_list',  default=[])
    parser.add_argument('--rounds', type=int, default=0)
    parser.add_argument('--model', default = None)
    parser.add_argument('--mode', type=str, default='solo')
    args = parser.parse_args()
    
    
    
    
    
    
    topk_numbs = [10,5]
    # topk_numbs = [10]
    model = None 
    acc_results = {}
    ref_model = None
    if args.mode == 'ref':
        ref_model = ref_model_training(args)
    
        
    
    for round in range(len(topk_numbs)):
        print("Round : ", round)
        model, now , epoch_acc= train(args, model, ref_model)
        top_func_idx_list = get_topk_funcs(model.alphas, topk_numbs[round])
        top_func_name_list = get_top_func_list_model(model, top_func_idx_list)

        # evaluate(model,args, now)
        acc_results[round] = epoch_acc        
        # prune! 
        for i in range(args.layer):
            model.layers[i].flist = remove_other_funcs(model.layers[i].flist, top_func_idx_list[i], args)

        # save model with pruned funcs
        torch.save(model.state_dict(), f'./result_model/model_{args.type}_{now}_round_{round}.pth')            
        args.rounds += 1
        model.layers[0].rounds = 1
    
    df = pd.DataFrame(top_func_name_list)
    df = df.transpose()
    df.to_csv(f'./result_csv/top_func_list_{args.type}_{args.batch_size}_{args.fig_name}.csv', mode='a', header=False, index=False)
    
    #final round
    model, now, epoch_acc = train(args, model, ref_model)
    acc_results[len(topk_numbs)] = epoch_acc
    #plot epoch accuracy
    visualize_epoch_acc(acc_results, now, args)    
    
