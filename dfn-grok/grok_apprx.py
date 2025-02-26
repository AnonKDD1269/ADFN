'''Code based on Neel Nanda's work on Grokking Transformers'''
'''https://github.com/neelnanda-io/Grokking'''
'''A Mechanistic Interpretability Analysis of Grokking(2023)'''

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
# from fancy_einsum import einsum
import os
from tqdm import tqdm 
import random
# from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import os 

from transformer_simple import Transformer
import matplotlib.pyplot as plt
# from utils import loss_fn

from make_dset import build_dataset
import argparse 
import pickle
from mixedfunc import MixedFunc
from func_prune import get_topk_funcs, get_top_func_list_model
import pandas as pd 
TRAIN_MODEL = True
device='cuda'

# """Plotting helper functions:"""

# def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
#     px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

# def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
#     px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
#     #save plot
#     plt.savefig('plot.png')
    
    

# def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
#     x = utils.to_numpy(x)
#     y = utils.to_numpy(y)
#     px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def loss_fn(logits, labels):
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()




"""## Define Task
* Define modular addition
* Define the dataset & labels

Input format:
|a|b|=|
"""




def train(dataset,labels,args):

    """Convert this to a train + test set - 30% in the training set"""
    DATA_SEED = args.DATA_SEED
    frac_train = args.frac_train
    p = args.p
    num_epochs = args.num_epochs
    lr = args.lr
    wd = args.wd
    betas = args.betas
    TRAIN_MODEL = args.TRAIN_MODEL
    device = args.device
    PTH_LOCATION = 'grok_trained.pth'

    # all the seedings
    torch.manual_seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    random.seed(DATA_SEED)
    torch.cuda.manual_seed(DATA_SEED)
    torch.cuda.manual_seed_all(DATA_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    
    
    
    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    test_data = dataset[test_indices]
    test_labels = labels[test_indices]



    """## Define Model"""

    cfg = HookedTransformerConfig(
        n_layers = 1,
        n_heads = 4,
        d_model = 128,
        d_head = 32,
        d_mlp = 512,
        act_fn = "relu",
        normalization_type=None,
        d_vocab=p+1,
        d_vocab_out=p,
        n_ctx=3,
        init_weights=True,
        device=device,
        seed = 999,
        use_hook_mlp_in = True
    )
    model = Transformer(cfg,args).to(device)
    def count_parameters(model):
        # display the summation progress.
        return sum(p.numel() for n,p in model.named_parameters() if p.requires_grad)
    print(f"My Model has {count_parameters(model)} parameters")
    exit()
    """Disable the biases, as we don't need them for this task and it makes things easier to interpret."""


    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    print(model)
    """## Define Optimizer + Loss"""

    my_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    print("Uniform loss:")
    print(np.log(p))


    """## Actually Train

    **Weird Decision:** Training the model with full batch training rather than stochastic gradient descent. We do this so to make training smoother and reduce the number of slingshots.
    """

    train_losses = []
    test_losses = []
    model_checkpoints = []
    checkpoint_epochs = []
    my_losses = []
    my_test_losses = []

    if args.from_pretrained:
        # model.load_state_dict(torch.load('base_model.pth'), strict=False)
        # load manually
        state_dict = torch.load('base_model_25000.pth')
        for name, param in model.named_parameters():
            if name in state_dict and "alpha" not in name and "beta" not in name:
                param.data = state_dict[name]


        with torch.no_grad():
            model.save_flag = True
            logits = model(train_data)
            test_out = model(test_data)
            print("Loaded model, test loss is", loss_fn(test_out, test_labels).item())
            return train_losses, test_losses, my_losses, my_test_losses, model, (test_data, test_labels)
        
        
    if TRAIN_MODEL:
        for epoch in tqdm(range(num_epochs)):
            if epoch+1 == num_epochs:
                model.save_flag = True

            logits = model(train_data)
            loss = loss_fn(logits, train_labels)
            loss.backward()
            my_optimizer.step()
            my_optimizer.zero_grad()
            my_losses.append(loss.item())

            with torch.inference_mode():
                # if model.save_flag:
                #     model.save_flag = False
                my_test_logits = model(test_data)
                my_test_loss = loss_fn(my_test_logits, test_labels)
                my_test_losses.append(my_test_loss.item())

    loss_df = pd.DataFrame(my_losses)
    loss_df.to_csv(f'loss_csv/losses_train_{DATA_SEED}.csv', index=False, header=False)
    test_loss_df = pd.DataFrame(my_test_losses)
    test_loss_df.to_csv(f'loss_csv/losses_test_{DATA_SEED}.csv', index=False, header=False)

    torch.save(
        {
            "model":model.state_dict(),
            "config": model.cfg,
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "test_losses": my_test_losses,
            "train_losses": my_losses,
            "train_indices": train_indices,
            "test_indices": test_indices,
        },
        PTH_LOCATION)

    return train_losses, test_losses, my_losses, my_test_losses, model, (test_data, test_labels)


def get_elbows(losses, labels):
    
    window = 1000
    # get the 2nd derivative of the losses, and find the index
    values = losses[window:]
    first_derivative = np.diff(values)
    second_derivative = np.diff(first_derivative)
    # get two elbows
    elbow1 = np.argmax(np.abs(second_derivative)) + window
    elbow2 = np.argmax(np.abs(second_derivative[elbow1+1:])) + elbow1 + 1 + window
    print(f"Elbow 1: {elbow1+window}, Elbow 2: {elbow2+window}, label: {labels}")
    # draw vertical line at the elbow
    plt.axvline(x=elbow1, color='g', linestyle='--')
    plt.axvline(x=elbow2, color='y', linestyle='--')


def force_beta_update(beta):
        # implement simple Sgd update for beta values.
        # 이거 층별로 안하고 있네
        
        for i in range(len(beta)):
            beta[i] = beta[i] - (beta[i].grad * 0.01)
def approximate_grok(model,args,val_test_data):

    train_data = model.mid_inputs[0].detach()
    train_labels = model.mid_outputs[0].detach()
    test_data = model.mid_inputs[1].detach()
    test_labels = model.mid_outputs[1].detach()
    
    loss_fn = nn.MSELoss()
    wd = args.wd
    betas = args.betas
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    pbar = tqdm(range(args.apprx_epochs))
    thousand_losses = []
    thousand_test_losses = []
    mean_losses = []
    mean_test_losses = []
    for epoch in pbar:
        optimizer.zero_grad()
        data_input = train_data.cuda()
        data_output = train_labels.cuda()
        output = model.apprx_forward(data_input)
        loss = loss_fn(output, data_output)
        loss.backward()
        # force beta update
        # force_beta_update(model.mixedfunc.beta)

        optimizer.step()



        with torch.inference_mode():
            test_data_input = test_data.cuda()
            test_data_output = test_labels.cuda()
            test_output = model.apprx_forward(test_data_input)
            test_loss = loss_fn(test_output, test_data_output)
        pbar.set_description(f"Loss: {loss.item():4f}, Test Loss: {test_loss.item():4f}")

        # every 1000 epochs, plot the loss and test loss
        # after 2000 epochs, do it
        if epoch >= 0:
            thousand_losses.append(loss.item())
            thousand_test_losses.append(test_loss.item())
            if epoch % 100 == 0:
                mean_losses.append(np.mean(thousand_losses))
                mean_test_losses.append(np.mean(thousand_test_losses))
            
                our_end_loss = test_grok(model, val_test_data[0], val_test_data[1])
                # save mean apprx loss and mean test loss
                df = pd.DataFrame([np.mean(thousand_losses)])
                df.to_csv('loss_csv/mean_apprx_losses.csv', index=False, mode='a', header=False)
                df = pd.DataFrame([np.mean(thousand_test_losses)])
                df.to_csv('loss_csv/mean_apprx_test_losses.csv', index=False, mode='a', header=False)
                # save end loss
                df = pd.DataFrame([our_end_loss])
                df.to_csv('loss_csv/end_apprx_test_losses.csv', index=False, mode='a', header=False)
                # plot the loss
                thousand_losses = []
                thousand_test_losses = []

                plt.plot(mean_losses, label='train')
                plt.plot(mean_test_losses, label='test')
                plt.legend()
                plt.savefig('train_test_loss_apprx.png')
                plt.clf()  


def compare_outputs(model, test_data, test_label):
    '''comparing the absolute magnitude of the outputs'''
    # using alternate forward
    test_data = test_data.detach()
    test_label = test_label.detach()
    test_data = test_data.cuda()
    test_label = test_label.cuda()

    with torch.inference_mode():
        test_output = model.alternate_forward(test_data)
        test_original = model(test_data)
    
    # plot the norm
    plt.plot(torch.mean(torch.norm(test_output, dim=1)).cpu().detach().numpy(), label='ours')
    plt.plot(torch.mean(torch.norm(test_original, dim=1)).cpu().detach().numpy(), label='original')
    plt.legend()
    plt.savefig('norm_comparison.png')

def test_grok(model, test_data, test_label):
    # using alternate forward
    test_data = test_data.detach()
    test_label = test_label.detach()
    test_data = test_data.cuda()
    test_label = test_label.cuda()

    with torch.inference_mode():
        test_output = model.alternate_forward(test_data)
        loss = loss_fn(test_output, test_label)
        test_original = model(test_data)
        test_original_loss = loss_fn(test_original, test_label)
    # save it to csv, test_original loss
    df = pd.DataFrame([test_original_loss.item()])
    df.to_csv('loss_csv/test_original_loss.csv', index=False, header=False)
    return loss.item()

def summarize_grok(model):
    # display alpha
    alpha = model.alphas
    print(f"Alpha: {alpha}")
    # funciton names are
    top_func_idx_list = get_topk_funcs(model.alphas,10)
    top_func_name_list = get_top_func_list_model(model, top_func_idx_list)
    alphas = [x for x in model.alphas[0]]
    top_alphas = [alphas[0][i] for i in top_func_idx_list]
    print(f"Top functions are {top_func_name_list}, top alphas are {top_alphas}")
    # consistency of alpha,by saving functions names tops 10, save it as csv, in mode "a"
    df = pd.DataFrame(columns=top_func_name_list)
    df.to_csv('top_func_consistency_check.csv', mode='a', index=False)
    
    # model, now = train(args,model)    
    # save flist to csv file in 'a' mode, single layer
    df = pd.DataFrame(top_func_name_list)
    # in same row
    df = df.T
    df.to_csv(f'./result_csv/flist_grok.csv', mode='a', header=False, index=False)
    
    # save alphas
    # df = pd.DataFrame([x.cpu().detach().numpy() for x in model.alphas[0]])
    # new df 7,27,1,8,10 only
    picked_alphas = [alphas[0][i].cpu().detach().numpy() for i in top_func_idx_list[0]]
    df = pd.DataFrame(picked_alphas)
    df.to_csv(f'./result_csv/alpha_grok.csv', mode='a', header=False, index=False)


if 'main' in __name__:
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--frac_train', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1.)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--num_epochs', type=int, default=25000)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--DATA_SEED', type=int, default=598)
    parser.add_argument('--TRAIN_MODEL', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')

    # dfn arguments
    parser.add_argument('--alpha_mult', type=int, default=1)
    parser.add_argument('--n_func', type=int, default=1)
    parser.add_argument('--soft_flag', type=int, default=0)
    parser.add_argument('--beta_alter', type=int, default=0)
    parser.add_argument('--beta_regul', type=float, default=0)
    parser.add_argument('--apprx_epochs', type=int, default=25000)
    parser.add_argument('--func_transfer', type=int, default=1)
    parser.add_argument('--from_pretrained', type=int, default=0)

    args = parser.parse_args()
    dataset,labels = build_dataset(args.p)
    train_losses, test_losses, my_losses, my_test_losses, model, test_data = train(dataset,labels,args)
    

    # # all the model
    # torch.save(model.state_dict(),f'base_model_{args.num_epochs}.pth')

    approximate_grok(model,args, test_data)
    test_grok(model, test_data[0], test_data[1])
    summarize_grok(model)
    compare_outputs(model, test_data[0], test_data[1])
    
    # plt.yscale('log')  # Set y-axis to log scale
    # # xticks at 2000
    # plt.xticks(np.arange(0, 25000, 2000))

    # plt.plot(my_losses, label='train_ours')
    # plt.plot(my_test_losses, label='test_ours')
    # plt.legend()

    # # get_elbows(my_losses, 'train_ours')
    # get_elbows(my_test_losses, 'test_ours')

    # plt.savefig('train_test_loss_test.png')
    # plt.clf()

