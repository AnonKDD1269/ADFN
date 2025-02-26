import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np
from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens.components import PosEmbed
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookedRootModule, HookPoint

import einops 
from mixedfunc import MixedFunc
from functions import FuncPool, FUNC_CLASSES
from func_manage import transfer_methods

# simple transformer architecture with single transformer block. 
# from scratch, without torch layers

class Transformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.n_layers  = cfg.n_layers
        self.n_heads   = cfg.n_heads
        self.d_model   = cfg.d_model
        self.d_head    = cfg.d_head
        self.d_mlp     = cfg.d_mlp
        self.act_fn    = cfg.act_fn
        self.normalization_type = cfg.normalization_type
        self.d_vocab   = cfg.d_vocab
        self.d_vocab_out = cfg.d_vocab_out
        self.n_ctx = cfg.n_ctx
        self.init_weight = True 
        self.device = cfg.device
        self.seed = cfg.seed
        self.mhattn =nn.ModuleList([MHA(self.d_model,self.d_head,self.n_heads,cfg) for _ in range(self.n_layers)]) # 66048
        self.unembed = UnEmbed(cfg) # 14577
        self.embed = Embed(cfg) # 14592
        self.positional_embedding = PosEmbed(cfg)
        self.cfg = cfg
        self.mlp = MLP(cfg) # 131712
        
        self.args = args

        # if self.init_weight:
        #     self.init_weights()
        
        # if self.args.func_transfer == True:
        
        #     class_keys = list(FUNC_CLASSES.keys())
        #     classes = [FUNC_CLASSES[key] for key in class_keys]
        #     # get initial func pool method list
        #     flist = [dir(FuncPool)[i] for i in range(len(dir(FuncPool))) if not dir(FuncPool)[i].startswith('_')]
        #     print("Initial len: ", len(flist))
        #     print("Initial flist: ", flist)
            
        #     # transfer methods from classes to FuncPool
        #     transfer_methods(classes, FuncPool)
        #     # get final func pool method list
        #     flist = [dir(FuncPool)[i] for i in range(len(dir(FuncPool))) if not dir(FuncPool)[i].startswith('_')]
        #     print("Transfer_res : ", len(flist))
        #     print("Transfered flist: ", flist)
        # # dfns
        # self.mid_inputs = []
        # self.mid_outputs = []
        # self.save_flag = False
        # self.flist= FuncPool()
        # self.n_func = 1
        # self.mixedfunc = MixedFunc(self.flist,args)

        # with torch.no_grad():
        #     self.alphas = self.initialize_alphas()
        #     dummy_input = torch.zeros(1, 3, self.d_model).to(self.device)
        #     out_lens = self.mixedfunc.forward_test(dummy_input)
        #     self.mixedfunc.initialize_betas_out(out_lens, args.p)
        #     self.mixedfunc.layer_idx = 0

        # # self.ones_weights = torch.ones(113,128,requires_grad=True).to(self.device)
        # #

            
    def alternate_forward(self, x):
        # positional embedding is done outside the model? no...
        x = self.embed(x)

        pemb = self.positional_embedding(x)
        x = x + pemb
        
        for i in range(self.n_layers):
            x = self.mhattn[i](x,x,x)
        
        x = self.mixedfunc(x, self.alphas)

        if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(
                        logits / self.cfg.output_logits_soft_cap
                    )
        return x

    def apprx_forward(self, x):

        x = self.mixedfunc(x, self.alphas)
        # print(self.alphas[0][0])
        # x = nn.functional.linear(x, self.ones_weights)
        return x


    
    def initialize_alphas(self):
        alpha = []
        '''mimic mixedop'''
        for i in range(self.n_func):
            # make enough alphas for sequence length
            # alpha.append(nn.Parameter(torch.ones(1, seq_len)/seq_len))
            alpha.append(torch.ones(1, len(self.mixedfunc.flist)*self.args.alpha_mult)
                         /len(self.mixedfunc.flist)*self.args.alpha_mult)
            # alpha.append(torch.randn(1, len(self.layers[0].flist)))
        alpha = nn.ParameterList(alpha)
        return alpha
    

    def forward(self, x):
        # positional embedding is done outside the model? no...
        x = self.embed(x)

        pemb = self.positional_embedding(x)
        x = x + pemb
        
        for i in range(self.n_layers):
            x = self.mhattn[i](x,x,x)
        
        if self.save_flag:
            self.mid_inputs.append(x)
        # in shape (batch, pos, d_model) (3830, 3, 128)
        x = self.mlp(x)
        x = self.unembed(x)
        # out shape (batch, pos, d_vocab_out) (3830, 3, 113)

        if self.save_flag:
            self.mid_outputs.append(x)
        if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(
                        logits / self.cfg.output_logits_soft_cap
                    )
        return x
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if "W_" in name or "weight" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)


class MHA(nn.Module):
    def __init__(self, d_model, d_head, n_heads,cfg):
        super().__init__()
        #multihead attention
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.dropout_rate = 0
        # self.attns = nn.ModuleList([BaseAttention(d_model,d_head,self.dropout_rate) for _ in range(n_heads)])
        self.attn = BaseAttention(d_model,d_head,n_heads,cfg)
    def forward(self,query,key,value):
        out = self.attn(query,key,value)
        return out

class BaseAttention(nn.Module):
    def __init__(self, d_model,d_head,n_heads, cfg):
        super().__init__()

        self.W_Q = nn.Parameter(torch.empty(n_heads,d_model, d_head), requires_grad=True)
        self.W_K = nn.Parameter(torch.empty(n_heads,d_model, d_head), requires_grad=True)
        self.W_V = nn.Parameter(torch.empty(n_heads,d_model, d_head), requires_grad=True)
        self.b_Q = nn.Parameter(torch.zeros(n_heads,d_head), requires_grad=True)
        self.b_K = nn.Parameter(torch.zeros(n_heads,d_head), requires_grad=True)
        self.b_V = nn.Parameter(torch.zeros(n_heads,d_head), requires_grad=True)
        self.W_O = nn.Parameter(torch.empty(n_heads,d_head,d_model), requires_grad=True)
        self.b_O = nn.Parameter(torch.zeros(d_model), requires_grad=True)
        self.dropout = nn.Dropout(0)
        self.scale_factor = np.sqrt(d_head)
        self.cfg = cfg

    def forward(self,query,key,value):
        qq = self._attn_linear(query, self.W_Q, self.b_Q)
        k = self._attn_linear(key, self.W_K, self.b_K)
        v = self._attn_linear(value, self.W_V, self.b_V)
        q_ = einops.rearrange(
            qq, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        k_ = einops.rearrange(
            k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        )

        attn_score = q_ @ k_ / self.scale_factor        
        pattern = torch.softmax(attn_score, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern).to("cuda")

        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        )
        pattern_ = einops.rearrange(
            pattern,
            "batch head_index query_pos key_pos -> batch head_index query_pos key_pos",
        )
        z = einops.rearrange(
                pattern_ @ v_,
                "batch head_index query_pos d_head -> batch query_pos head_index d_head",
                )

        w = einops.rearrange(
                self.W_O, "head_index d_head d_model -> d_model (head_index d_head)"
                )
        out = F.linear(
            z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads),
            w,
            self.b_O,
        )

        return out
    
    def _attn_linear(self, input, w, b):
        w = einops.rearrange(w, "head_index d_model d_head -> (head_index d_head) d_model")
        b_ = einops.rearrange(b, "head_index d_head -> (head_index d_head)")
        return F.linear(input, w, b_).reshape(input.shape[0], input.shape[1], b.shape[0], b.shape[1])
        
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=self.cfg.dtype, device=self.cfg.device))
     
    def forward(self, tokens):
        return self.W_E[tokens, :]

class UnEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_vocab_out, dtype=self.cfg.dtype, device=self.cfg.device))
        self.b_U = nn.Parameter(torch.zeros(self.cfg.d_vocab_out, dtype=self.cfg.dtype))
    def forward(self, residual):
        return batch_addmm(self.b_U, self.W_U, residual)


class MLP(CanBeUsedAsMLP):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.select_activation_function()

        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.d_mlp, dtype=self.cfg.dtype), requires_grad=True)
        self.b_in = nn.Parameter(torch.zeros(self.d_mlp, dtype=self.cfg.dtype), requires_grad=True)

        self.W_out = nn.Parameter(torch.empty(self.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))


    def forward(self, x):
        # This is equivalent to (roughly) W_in @ x + b_in. It's important to
        # use a fused addmm to ensure it matches the Huggingface implementation
        # exactly.
        pre_act = batch_addmm(self.b_in, self.W_in, x)  # [batch, pos, d_mlp]

        if (
            self.cfg.is_layer_norm_activation()
            and self.hook_mid is not None
            and self.ln is not None
        ):
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        else:
            post_act = self.act_fn(pre_act) # [batch, pos, d_mlp]
        return batch_addmm(self.b_out, self.W_out, post_act)
