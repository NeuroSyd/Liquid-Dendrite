import sys

print(sys.version)

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np

gamma = .5  # gradient scale
lens = 0.3

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = .15
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight

        return grad_input * temp.float() * gamma
        # return grad_input

act_fun_adp = ActFun_adp.apply

def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    alpha = tau_m

    ro = tau_adp

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = 0.1 + beta * b
    # B = 1.

    d_mem = -mem + inputs
    mem = mem + d_mem*alpha
    inputs_ = mem - B

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1-spike)*mem
    #ojo, one is b and the other B.
    return mem, spike, B

#%%
###############################################################################################
class sigmoid_beta(nn.Module):

    def __init__(self, alpha = 1.):
        super(sigmoid_beta,self).__init__()

        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(1.)) # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha

        self.alpha.requiresGrad = False # set requiresGrad to true!
        # self.alpha=alpha

    def forward(self, x):
        if (self.alpha == 0.0):
            return x
        else:
            return torch.sigmoid(self.alpha*x)

class SNN_rec_cell(nn.Module):
    def __init__(self, input_size, hidden_size,is_rec = True):
        super(SNN_rec_cell, self).__init__()
        # print('SNN-ltc ')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_rec = is_rec

        if is_rec:
            self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
        else:
            self.layer1_x = nn.Linear(input_size, hidden_size)

        self.layer1_tauM = nn.Linear(2*hidden_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(2*hidden_size, hidden_size)

        self.act1 = nn.Sigmoid()

        nn.init.xavier_uniform_(self.layer1_x.weight)
        nn.init.xavier_uniform_(self.layer1_tauM.weight)
        nn.init.xavier_uniform_(self.layer1_tauAdp.weight)

    def forward(self, x_t, mem_t,spk_t,b_t):

        if self.is_rec:
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)

        tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x,mem_t),dim=-1)))
        tauAdp1 = self.act1(self.layer1_tauAdp(torch.cat((dense_x,b_t),dim=-1)))

        mem_1,spk_1,b_1 = mem_update_adp(dense_x, mem=mem_t,spike=spk_t, b=b_t, tau_m=tauM1, tau_adp=tauAdp1)

        #tauM1 is alpha.
        #tauAdp1 is beta?
        #b1 is threshold?

        return mem_1,spk_1,b_1, tauM1, tauAdp1

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new_zeros(bsz, self.hidden_size),
                weight.new_zeros(bsz, self.hidden_size),
                weight.new_zeros(bsz, self.hidden_size),)

    def compute_output_size(self):
        return [self.hidden_size]

