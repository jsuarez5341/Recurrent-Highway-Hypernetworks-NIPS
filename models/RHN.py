from pdb import set_trace as T
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
  
def highwayGate(Ws, s, gateDrop, trainable):
   h = int(Ws.size()[1]/2)
   hh, tt  = t.split(Ws, h, 1)
   hh, tt = F.tanh(hh), F.sigmoid(tt) 
   cc = 1 - tt
   tt = F.dropout(tt, p=gateDrop, training=trainable)
   return hh*tt + s*cc

class RHNCell(nn.Module):
   def __init__(self, embedDim, h, depth, gateDrop):
      super(RHNCell, self).__init__()
      self.h, self.depth, self.gateDrop = h, depth, gateDrop
      self.inp  = nn.Linear(embedDim, 2*h)
      self.cell = nn.ModuleList([
            nn.Linear(h, 2*h) for i in range(depth)])

   def forward(self, x, s, trainable):
      if s is None:
         batchSz = x.size(0)
         s = [Variable(t.zeros(batchSz, self.h).cuda()) 
            for i in range(self.depth)]

      for l in range(self.depth):
         Ws = self.cell[l](s[l])
         if l == 0:
            Ws += self.inp(x) 
         s[l]  = highwayGate(Ws, s[l], self.gateDrop, trainable)
      return s[-1], s

class RHNNetwork(nn.Module):

   def __init__(self):
      super(RHNNetwork, self).__init__()
      self.rhnInp = RHNCell(embedDim, hNetwork, hNetwork)
      if depth > 1:
         self.rhn = utils.list(RHNCell, hNetwork, hNetwork, hNetwork, n=depth-1)
      self.depth = depth

   def forward(self, x, trainable):
      states = [Variable(torch.zeros(batchSz, hNetwork).cuda()) 
            for i in range(depth)]

      out = []
      for i in range(context):
         h = self.rhnInp(x[:, i], states[0], trainable)
         states[0] = h
         if self.depth > 1:
            for l in range(depth-1):
               h = self.rhn[l](h, states[l+1], trainable)
               states[l+1] = h
         out += [h]
         
      return out
