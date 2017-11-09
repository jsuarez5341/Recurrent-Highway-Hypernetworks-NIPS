import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from pdb import set_trace as T

class HyperLinear(Module):
    def __init__(self, in_features, out_features):
        super(HyperLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, z):
        weight = self.weight
        z = torch.cat((z,z), 1)
        Wx = self._backend.Linear()(input, weight)*z
        Wx += self.bias.expand_as(Wx) 
        return Wx

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
