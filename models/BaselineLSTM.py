from pdb import set_trace as T
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class LSTMCellLayer(nn.Module):

   def __init__(self, embedDim, h, gateDrop):
      super(LSTMCellLayer, self).__init__()
      self.h, self.gateDrop = h, gateDrop
      self.fc = nn.Linear(embedDim+h, 4*h)

   def forward(self, x, s, trainable):
      hh, c = s
      xh = t.cat((x, hh), 1)

      Wx = self.fc(xh)
      i, f, o, g = t.split(Wx, self.h, 1)

      i = F.sigmoid(i)
      f = F.sigmoid(f)
      o = F.sigmoid(o)
      g = F.tanh(g)

      if trainable:
         g = F.dropout(g, p=self.gateDrop, training=trainable)

      c = c*f + g*i
      hh = F.tanh(c) * o

      hc = (hh, c)
      return hh, hc

class LSTMCell(nn.Module):
   def __init__(self, embedDim, h, depth, gateDrop):
      super(LSTMCell, self).__init__()
      self.h, self.depth = h, depth
      self.lstmInp = LSTMCellLayer(embedDim, h, gateDrop)
      self.lstm = nn.ModuleList([
            LSTMCellLayer(h, h, gateDrop) for i in range(depth-1)])

   def initializeIfNone(self, s, batchSz):
      if s is not None: return s
      return [[
            Variable(t.zeros(batchSz, self.h).cuda()),
            Variable(t.zeros(batchSz, self.h).cuda())]
            for i in range(self.depth)]

   def forward(self, x, s, trainable):
      s, sNew = self.initializeIfNone(s, x.size(0)), []
      out, ss = self.lstmInp(x, s[0], trainable)
      sNew += [ss]
      for l in range(self.depth-1):
         out, ss = self.lstm[l](out, s[l+1], trainable)
         sNew += [ss]

      return out, sNew
