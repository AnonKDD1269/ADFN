from data_factory import data_provider
from exp_basic import Exp_Basic
import DLinear
from tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from metrics import metric
from DDDFN import DFNModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm 
import os
import time
from functions import FuncPool, FUNC_CLASSES
import warnings
import matplotlib.pyplot as plt
import numpy as np
from func_manage import transfer_methods
from func_prune import get_topk_funcs, remove_other_funcs


class Exp_DFN(Exp_Basic):
    def __init__(self, args):
        super(Exp_DFN, self).__init__(args)
    
    def _build_model(self):
        
        if self.args.func_transfer == True:
        
            class_keys = list(FUNC_CLASSES.keys())
            classes = [FUNC_CLASSES[key] for key in class_keys]
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

        
        self.model = DFNModel(n_func=self.args.n_func, in_length=self.args.seq_len, pred_len=self.args.pred_len)
        self.model = self.model.to(self.device)

        # do forward test
        self.model.mixedfunc_seasonal.forward_test(torch.randn(1, 1, self.args.seq_len).to(self.device))
        self.model.mixedFunc_trend.forward_test(torch.randn(1, 1, self.args.seq_len).to(self.device))


        print('Model Structure: ', self.model)
        return self.model
    

    def _get_data(self,flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train_round(self,model,setting):
        
        train_data, train_loader = self._get_data(flag='train')


        # test and get out_lens for mixedfunc
        out_lens = self.model.general_forward_test(torch.randn(1, 1, self.args.seq_len).to(self.device))

        self.model.layers[0].initialize_betas_out(out_lens[0], self.args.pred_len)
        self.model.layers[1].initialize_betas_out(out_lens[1], self.args.pred_len)

        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')
            
        path = os.path.join(self.args.save_path, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._get_optimizer()
        criterion = self._select_criterion()
        
        epoch_pbar = tqdm(range(self.args.train_epochs))
        aux_loss = self._select_criterion()

        for epoch in epoch_pbar:
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(enumerate(train_loader),leave=False, total=train_steps)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
            print('Epoch: ', epoch, 'Training Loss: ', np.mean(train_loss), 'Time: ', time.time()-epoch_time)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model 
    
    def get_top_func_list_model(self,model,top_idx_list):
        funclist = {}
        for i in range(2):
            funclist[i] = model.layers[i].get_top_func_names_layer(top_idx_list[i])

        return funclist   
    


    def train(self,setting):
        
        topk_lists = [17, 10, 5]

        for round in range(len(topk_lists)):
            self.model = self.train_round(self.model,setting)
            top_func_idx_list_seasonal = get_topk_funcs(self.model.alphas, topk_lists[round])
            top_func_idx_list_trend = get_topk_funcs(self.model.alphas_2, topk_lists[round])
            top_Func_idx_list_both = [top_func_idx_list_seasonal, top_func_idx_list_trend]
            top_func_name_list = self.get_top_func_list_model(self.model, top_Func_idx_list_both)

            # prune now
            self.model.layers[0].flist = remove_other_funcs(self.model.layers[0].flist, top_func_idx_list_seasonal, self.args)
            self.model.layers[1].flist = remove_other_funcs(self.model.layers[1].flist, top_func_idx_list_trend, self.args)

            # reinitialize alphas
            self.model.alphas = self.model.initialize_alphas()
            self.model.alphas_2 = self.model.initialize_alphas()
            self.model.update_funcs()
            
            # self.model.alphas = self.model.initialize_alphas()
            # self.model.alphas_2 = self.model.initialize_alphas()
            # self.model.update_funcs()
        print("Last round, top func names: ", top_func_name_list)
        print("begin pruned training...")
        out_lens = self.model.general_forward_test(torch.randn(1, 1, self.args.seq_len).to(self.device))

        self.model.layers[0].initialize_betas_out(out_lens[0], self.args.pred_len)
        self.model.layers[1].initialize_betas_out(out_lens[1], self.args.pred_len)

        self.model = self.train_round(self.model,setting)
        
        
        


        return self.model
    

    def test(self, setting, test=0):
        
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.apprx:
                    breakpoint()
                    outputs = self.model.alternative_forward(batch_x, self.model.alphas) 
                else:
                    breakpoint()
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        # print final alpha of both layer
        print('Final alpha 1: ', self.model.alphas[0].detach().cpu().numpy())
        print('Final alpha 2: ', self.model.alphas_2[0].detach().cpu().numpy())