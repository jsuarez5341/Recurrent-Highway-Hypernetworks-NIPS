from pdb import set_trace as T
import sys, shutil
import time
import numpy as np
import torch
import torch as t
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable 
import utils
from params import *

from oldRawLangBatcher import RawBatcher
from HyperLinear import HyperLinear

def highwayGate(Ws, s, trainable):
   h = int(Ws.size()[1]/2)
   hh, tt  = t.split(Ws, h, 1)
   hh, tt = F.tanh(hh), F.sigmoid(tt) 
   cc = 1 - tt
   tt = F.dropout(tt, p=gateDrop, training=trainable)
   return hh*tt + s*cc

class HyperRHN(nn.Module):

   def __init__(self):
      super(HyperRHN, self).__init__()
      #Hypernet
      self.hiddenHyper = utils.list(nn.Linear, hHyper, 2*hHyper, n=depth)
      self.inputHyper  = nn.Linear(embedDim, 2*hHyper)

      #RHN
      self.hidden = utils.list(HyperLinear, hNetwork, 2*hNetwork, n=depth)
      self.input = HyperLinear(embedDim, 2*hNetwork)

      self.upscaleProj = utils.list(nn.Linear, hHyper, hNetwork, n=depth)

   def forward(self, x, sNetwork, sHyper, trainable):
      for i in range(depth):
         #Hypernet
         Ws = self.hiddenHyper[i](sHyper) if sHyper != 0 else 0
         if i == 0:
            Ws += self.inputHyper(x)
         sHyper = highwayGate(Ws, sHyper, trainable)

         #Upscale
         z = self.upscaleProj[i](sHyper)

         #RHN
         Ws = self.hidden[i](sNetwork, z) if sNetwork != 0 else 0
         if i == 0:
            Ws += self.input(x, z)
         sNetwork= highwayGate(Ws, sNetwork, trainable)

      return sHyper, sNetwork

class Network(nn.Module):

   def __init__(self):
      super(Network, self).__init__()
      self.HyperRHN = HyperRHN()

      self.embed   = nn.Embedding(vocab, embedDim)
      self.unembed = nn.Linear(hNetwork, vocab)
      #self.network = network()

   def forward(self, x, trainable):
      x = self.embed(x)
      x = F.dropout(x, p=embedDrop, training=trainable)

      sHyper, sNetwork = 0, 0 #RHN will handle initialization
      out = []

      for i in range(context):
         sHyper, sNetwork = self.HyperRHN(x[:, i], 
               sNetwork, sHyper, trainable)
         out += [sNetwork]
         
      x = t.stack(out, 1)
      x = x.view(batchSz*context, hNetwork)
      x = self.unembed(x)
      x = x.view(batchSz, context, vocab)
      return x
 
class LSTMNetwork(nn.Module):

   def __init__(self):
      super(LSTMNetwork, self).__init__()
      self.lstmInp = nn.LSTMCell(embedDim, hNetwork)
      self.lstm = utils.list(nn.LSTMCell, hNetwork, hNetwork, n=depth-1)

   def forward(self, x, trainable):
      states = [[
            Variable(torch.zeros(batchSz, hNetwork).cuda()), 
            Variable(torch.zeros(batchSz, hNetwork).cuda())]
            for i in range(depth)]

      out = []
      for i in range(context):
         h, c = self.lstmInp(x[:, i], states[0])
         for l in range(depth-1):
            h, c = self.lstm[l](h, states[l+1])
            states[l+1] = [h, c]
         out += [h]
         
      return out

class EmbedUnembedWrapper(nn.Module):
   
   def __init__(self, network):
      super(EmbedUnembedWrapper, self).__init__()
      self.embed   = nn.Embedding(vocab, embedDim)
      self.unembed = nn.Linear(hNetwork, vocab)
      self.network = network()

   def forward(self, x, trainable):
      x = self.embed(x)
      x = F.dropout(x, p=embedDrop, training=trainable)

      x = self.network(x, trainable)

      x = t.stack(x, 1)
      x = x.view(batchSz*context, hNetwork)
      x = self.unembed(x)
      x = x.view(batchSz, context, vocab)
      return x

#class Network(nn.Module):
   
#   def __init__(self, network):
#      super(Network, self).__init__()
#      self.net = EmbedUnembedWrapper(network)

#   def forward(self, x, trainable):
#      return self.net(x, trainable)

