from pdb import set_trace as T
import torch as t
import torch.nn.functional as F
from torch import nn
  
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
      sOut = []
      for l in range(self.depth):
         Ws = self.cell[l](s) if s is not 0 else 0
         if l == 0:
            Ws += self.inp(x) 
         s = highwayGate(Ws, s, self.gateDrop, trainable)
         sOut += [s]
      return s, s, sOut
