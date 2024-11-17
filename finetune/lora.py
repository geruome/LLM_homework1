import torch
import transformers

from utils import recursive_getattr, recursive_setattr
from pdb import set_trace as stx
import time
import torch.nn as nn
import math
import torch.nn.functional as F

time1 = 0; cnt = 0; pre_time = 0

class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling): # lora_dim: r
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        out_features, in_features = weight.shape
        self.lora_right_weight = torch.nn.Parameter(torch.zeros((in_features, lora_dim))) # matrix A
        self.lora_left_weight = torch.nn.Parameter(torch.zeros((lora_dim, out_features))) # matrix B
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        # nn.init.normal_(self.lora_right_weight, std=0.01)

    # def forward(self, input): # (16, 139, 768)
    #     w = self.lora_right_weight @ self.lora_left_weight * self.lora_scaling
    #     w = self.weight + w.t()
    #     res = F.linear(input, w, bias=self.bias)
    #     return res

    def forward(self, input): # (16, 139, 768)
        res = F.linear(input, self.weight, bias=self.bias)
        lora_term = (input @ self.lora_right_weight) @ self.lora_left_weight * self.lora_scaling
        res += lora_term
        return res

def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_right_weight" in name or "lora_left_weight" in name
    return model

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            lora_state_dict[name] = param.data.clone()
    return lora_state_dict