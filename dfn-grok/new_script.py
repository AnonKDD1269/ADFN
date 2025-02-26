import torch 
from transformer_lens.components import PosEmbed
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from transformer_simple import Transformer

device = 'cuda'
p= 113
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
)

model = Transformer(cfg)

print(model.embed.W_E)
print(model.unembed.W_U)