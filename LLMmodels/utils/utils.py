import torch
import numpy as np
import random
import os
from LLMmodels.model.gpt_model import *
from LLMmodels.model.llama_model import *

def get_LLM_network(cfg):
    if 'gpt' in cfg.model_name.lower():
        net = GPT(cfg)
    elif 'llama' in cfg.model_name.lower():
        net = LlamaModel(cfg)
    else:
        print(f"Unexpected model_name: {cfg.model_name}")
    return net


def print_paramwise_parameters(model):
    seen_params = set()
    for name, param in model.named_parameters():
        if id(param) not in seen_params:
            seen_params.add(id(param))
            print(f"Parameter: {name} | Size: {param.size()} | Total Elements: {param.numel()}")
        else:
            print(f"Parameter: {name} | Size: {param.size()} | Total Elements: {param.numel()} | Duplicate")
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    return 
