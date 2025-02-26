# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import os
from tqdm import tqdm 
import random
from pathlib import Path
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
from utils import loss_fn
from make_dset import build_dataset
import argparse 


TRAIN_MODEL = True
device='cuda'

"""Plotting helper functions:"""

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
    #save plot
    plt.savefig('plot.png')
    
    

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


PTH_LOCATION = "./grok.pth"


# """# Model Training

# ## Config
# """

# p = 113
# frac_train = 0.3

# # Optimizer config
# lr = 1e-3
# wd = 1.
# betas = (0.9, 0.98)

# num_epochs = 25000
# checkpoint_every = 100

# DATA_SEED = 598



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

    torch.manual_seed(DATA_SEED)
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
    model = HookedTransformer(cfg)
    my_model = Transformer(cfg,args).to(device)
    
    def count_parameters(model):
        # display the summation progress.
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {count_parameters(model)} parameters")
    print(f"My Model has {count_parameters(my_model)} parameters")

    """Disable the biases, as we don't need them for this task and it makes things easier to interpret."""

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    for name, param in my_model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    print(model)
    print("#################")
    print(my_model)
    """## Define Optimizer + Loss"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    my_optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, weight_decay=wd, betas=betas)

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
    if TRAIN_MODEL:
        for epoch in tqdm(range(num_epochs)):
            # train_logits = model(train_data)
            # train_loss = loss_fn(train_logits, train_labels)
            # train_loss.backward()
            # train_losses.append(train_loss.item())
            # optimizer.step()
            # optimizer.zero_grad()

            my_logits = my_model(train_data)
            my_loss = loss_fn(my_logits, train_labels)
            my_loss.backward()
            my_optimizer.step()
            my_optimizer.zero_grad()
            my_losses.append(my_loss.item())


            # with torch.inference_mode():
            #     test_logits = model(test_data)
            #     test_loss = loss_fn(test_logits, test_labels)
            #     test_losses.append(test_loss.item())

            with torch.inference_mode():
                my_test_logits = my_model(test_data)
                my_test_loss = loss_fn(my_test_logits, test_labels)
                my_test_losses.append(my_test_loss.item())

    #         if ((epoch+1)%checkpoint_every)==0:
    #             checkpoint_epochs.append(epoch)
    #             model_checkpoints.append(copy.deepcopy(model.state_dict()))
    #             print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")
    #             print(f"Epoch {epoch} My Train Loss {my_loss} Test Loss {my_test_loss}")

    # torch.save(
    #     {
    #         "model":model.state_dict(),
    #         "config": model.cfg,
    #         "checkpoints": model_checkpoints,
    #         "checkpoint_epochs": checkpoint_epochs,
    #         "test_losses": test_losses,
    #         "train_losses": train_losses,
    #         "train_indices": train_indices,
    #         "test_indices": test_indices,
    #     },
    #     PTH_LOCATION)

    return train_losses, test_losses, my_losses, my_test_losses

def approximate(dataset,labels,args):
    if not TRAIN_MODEL:
        cached_data = torch.load(PTH_LOCATION)
        # model.load_state_dict(cached_data['model'])
        model_checkpoints = cached_data["checkpoints"]
        checkpoint_epochs = cached_data["checkpoint_epochs"]
        test_losses = cached_data['test_losses']
        train_losses = cached_data['train_losses']
        train_indices = cached_data["train_indices"]
        test_indices = cached_data["test_indices"]

    # from neel_plotly.plot import line
    # line([train_losses[::100], test_losses[::100]], 
    #     x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss",
    #     log_y=True, title="Training Curve for Modular Addition", 
    #     line_labels=['train', 'test'], toggle_x=True, toggle_y=True)

    plt.plot(train_losses, label='train_original')
    plt.plot(test_losses, label='test_original')


    # draw mine with plt
    plt.plot(my_losses, label='train_ours')
    plt.plot(my_test_losses, label='test_ours')

    plt.yscale('log')  # Set y-axis to log scale

    plt.legend()
    plt.savefig('train_test_loss.png')

def get_elbows(losses, labels):
    
    window = 0
    # get the 2nd derivative of the losses, and find the index
    values = losses[1000:2000]
    first_derivative = np.diff(values)
    second_derivative = np.diff(first_derivative)
    # get two elbows
    elbow1 = np.argmax(np.abs(second_derivative))
    elbow2 = np.argmax(np.abs(second_derivative[elbow1+1:])) + elbow1 + 1
    print(f"Elbow 1: {elbow1+window}, Elbow 2: {elbow2+window}, label: {labels}")
    # draw vertical line at the elbow
    plt.axvline(x=elbow1, color='r', linestyle='--')
    plt.axvline(x=elbow2, color='b', linestyle='--')

    



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

    args = parser.parse_args()
    dataset,labels = build_dataset(args.p)
    train_losses, test_losses, my_losses, my_test_losses = train(dataset,labels,args)
    # draw with plt
    # plt.plot(train_losses, label='train_oris')    
    # plt.plot(test_losses, label='test_oris')
    plt.yscale('log')  # Set y-axis to log scale
    # xticks at 1000
    plt.xticks(np.arange(0, 25000, 2000))

    plt.plot(my_losses, label='train_ours')
    plt.plot(my_test_losses, label='test_ours')
    plt.legend()

    # get the 2nd derivative of the losses, and find the index
    # get_elbows(train_losses, 'train_oris')
    # get_elbows(test_losses, 'test_oris')

    get_elbows(my_losses, 'train_ours')
    get_elbows(my_test_losses, 'test_ours')

    plt.savefig('train_test_loss.png')

