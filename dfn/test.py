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
import argparse


def test_pipe(args):
    
    
    model = FuncControlModel(args.layer)
    
    dummy_input = torch.ones(1,1).float()
    out_length = model.forward_test(dummy_input)
    max_length = max(out_length)
    print("did start")
    # get func length and arg length. 
    model.initialize_betas(out_length,max_length)
    model.load_state_dict(torch.load(args.model_path))
    
    model.eval()
    model.cuda()
    dummy_input = torch.ones(2,1).float().cuda()
    a = args.inp[0]
    b = args.inp[1]
    
    mod = 5
    
    x = torch.tensor([a,b]).float().cuda()
    x = x.unsqueeze(dim=0)
    print("input : ", x)
    out = model(x)
    print("output : ", out)
    print("answer is : ", torch.div((torch.sum(x, dim=1) % mod).float(), 100)) 
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, help='model path', default='model_modular.pth')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--layer', type=int, default=1 )
    parser.add_argument('--type', type=str, default='sin')
    parser.add_argument('--fig_name', type=str, default='figname')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--func_transfer', type=int, default=1)
    parser.add_argument('--inp', type=list, default=[7,2])
    args = parser.parse_args()
    test_pipe(args)
    