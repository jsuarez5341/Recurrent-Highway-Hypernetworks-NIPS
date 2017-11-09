from pdb import set_trace as T
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from HyperLinear import HyperLinear
from models.RHN import RHNCell
from models.RHN import highwayGate 

class HyperCell(nn.Module):
   def __init__(self, embedDim, h, depth, gateDrop):
      super(HyperCell, self).__init__()
      self.h, self.depth, self.gateDrop = h, depth, gateDrop
      self.inp  = HyperLinear(embedDim, 2*h)
      self.cell = nn.ModuleList([
            HyperLinear(h, 2*h) for i in range(depth)])

   def initialzeIfNone(self, s, batchSz):
      if s is not None: return s
      return [Variable(t.zeros(batchSz, self.h).cuda())
         for i in range(self.depth)]

   def forward(self, x, s, z, trainable):
      s = self.initialzeIfNone(s, x.size(0))
      for l in range(self.depth):
         Ws = self.cell[l](s[l], z[l])
         if l == 0:
            Ws += self.inp(x, z[l])
         s[l]  = highwayGate(Ws, s[l], self.gateDrop, trainable)
      return s[-1], s

class HyperRHNCell(nn.Module):

   def __init__(self, embedDim, h, depth, gateDrop):
      super(HyperRHNCell, self).__init__()
      hHyper, hNetwork = h
      self.HyperCell = RHNCell(embedDim, hHyper, depth, gateDrop)
      self.RHNCell   = HyperCell(embedDim, hNetwork, depth, gateDrop)
      self.upscaleProj = nn.ModuleList([nn.Linear(hHyper, hNetwork) 
            for i in range(depth)])

   def initialzeIfNone(self, s):
      if s is not None: return s
      return (None, None)
 
   def forward(self, x, s, trainable):
      sHyper, sNetwork = self.initialzeIfNone(s)
      _, sHyper   = self.HyperCell(x, sHyper, trainable)
      z = [self.upscaleProj[i](e) for i, e in enumerate(sHyper)]
      out, sNetwork = self.RHNCell(x, sNetwork, z, trainable)
      return out, (sHyper, sNetwork)
