import torch

import torch.nn as nn
import torch.nn.functional as F


class TernConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TernConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.eps = 0.001

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        
        alpha = self.weight.mean()
        ternary_weights = self.ternarize_weights(self.weight - alpha)
        print(ternary_weights)
        ternary_input = self.tern_quantize_activations_groupwise(input)
        return F.conv2d(ternary_input, ternary_weights)
    

    def ternarize_weights(self,w):
        tern_weight = torch.zeros_like(w)
        tern_weight[w > 0.05] = 1
        tern_weight[w < -0.05] = -1
        return tern_weight

    def tern_quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x
