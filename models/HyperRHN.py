from pdb import set_trace as T
import torch as t
from torch import nn

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

   def forward(self, x, s, z, trainable):
      sOut = []
      for l in range(self.depth):
         Ws = self.cell[l](s, z[l]) if s is not 0 else 0
         if l == 0:
            Ws += self.inp(x, z[l])
         s = highwayGate(Ws, s, self.gateDrop, trainable)
         sOut += [s]
      return s, sOut

class HyperRHNCell(nn.Module):

   def __init__(self, embedDim, h, depth, gateDrop):
      super(HyperRHNCell, self).__init__()
      hHyper, hNetwork = h
      self.HyperCell = RHNCell(embedDim, hHyper, depth, gateDrop)
      self.RHNCell   = HyperCell(embedDim, hNetwork, depth, gateDrop)
      self.upscaleProj = nn.ModuleList([nn.Linear(hHyper, hNetwork) 
            for i in range(depth)])

   def initializeIfNone(self, s):
      if s is not 0: return s
      return (0, 0)
 
   def forward(self, x, s, trainable):
      sHyper, sNetwork = self.initializeIfNone(s)
      _, _, sHyper = self.HyperCell(x, sHyper, trainable)
      z = [self.upscaleProj[i](e) for i, e in enumerate(sHyper)]
      out, sNetwork = self.RHNCell(x, sNetwork, z, trainable)
      return out, (sHyper[-1], sNetwork[-1]), (sHyper, sNetwork)
