from mmcv.cnn import kaiming_init
import torch.nn as nn
import torch
import torch.nn.init as init
import math

def init_parameters_uniform(module, mode="fan_out", nonlinearity="relu", zerobias=True):
    a = None    
    if nonlinearity == 'leaky_relu' or nonlinearity == 'reset':
        nonlinearity = 'leaky_relu'
        zerobias = False
        mode = "fan_in"
        a = math.sqrt(5)
            
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(module.weight)
    fan = fan_in if mode == 'fan_in' else fan_out

    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    init.uniform_(module.weight, -bound, bound)

    if hasattr(module, 'bias') and module.bias is not None:
        if zerobias:
            init.constant_(module.bias, 0.0)
        else:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(module.bias, -bound, bound)

def init_parameters_normal(module, mode="fan_out", nonlinearity="relu", zerobias=True):
    a = None    
    if nonlinearity == 'leaky_relu' or nonlinearity == 'reset':
        nonlinearity = 'leaky_relu'
        zerobias = False
        mode = "fan_in"
        a = math.sqrt(5)

    fan_in, fan_out = init._calculate_fan_in_and_fan_out(module.weight)
    fan = fan_in if mode == 'fan_in' else fan_out

    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    init.trunc_normal_(module.weight, mean=0.0, std=std)

    if hasattr(module, 'bias') and module.bias is not None:
        if zerobias:
            init.constant_(module.bias, 0.0)
        else:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(module.bias, -bound, bound)

def init_parameters(module, distribution='uniform', mode="fan_out", nonlinearity="relu", zerobias=True):
    with torch.no_grad():
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            return init_parameters_uniform(module, mode, nonlinearity, zerobias)
        else:
            return init_parameters_normal(module, mode, nonlinearity, zerobias)

def init_constant(module, a=1.0, b=0.0):
    with torch.no_grad():
        if hasattr(module, 'weight') and module.bias is not None:
            init.constant_(module.weight, a)

        if hasattr(module, 'bias') and module.bias is not None:
            init.constant_(module.bias, b)