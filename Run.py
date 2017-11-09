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

from LanguageBatcher import LanguageBatcher
from HyperLinear import HyperLinear

#Load PTB
def dataBatcher(batchSz, context, minContext):
   print('Loading Data...')
   train = 'data/ptb.train.txt'
   valid = 'data/ptb.valid.txt'
   test  = 'data/ptb.test.txt'
   vocab = 'data/vocab.txt'

   trainBatcher = LanguageBatcher(train, vocab, 
         batchSz, context, 0, rand=True)
   validBatcher = LanguageBatcher(valid, vocab, 
         batchSz, context, minContext)
   testBatcher  = LanguageBatcher(test,  vocab, 
         batchSz, context, minContext)

   print('Data Loaded.')
   return trainBatcher, validBatcher, testBatcher

class Network(nn.Module):

   def __init__(self, cell, vocabDim, embedDim, 
         unembedDim, ansDim, context, embedDrop):
      super(Network, self).__init__()
      self.cell, self.context, self.drop = cell, context, embedDrop
      self.embed   = nn.Embedding(vocabDim, embedDim)
      self.unembed = nn.Linear(unembedDim, ansDim)

   def forward(self, x, trainable):
      x, s, out = self.embed(x), None, []
      x = F.dropout(x, p=self.drop, training=trainable)

      for i in range(self.context):
         o, s = self.cell(x[:, i], s, trainable)
         out += [o]

      batchSz = x.size(0)
      x = self.unembed(t.cat(out, 0)).view(batchSz, self.context, -1)
      return x


def train(net, opt, trainBatcher, validBatcher, saver, minContext):
   while True:
      start = time.time()

      trainLoss, trainAcc = utils.runData(net, opt, trainBatcher,
            trainable=True, verbose=True)
      validLoss, validAcc = utils.runData(net, opt, validBatcher,
            minContext=minContext)

      trainLoss, validLoss = np.exp(trainLoss), np.exp(validLoss)

      print('\nEpoch: ', saver.epoch(), ', Time: ', time.time()-start)
      print('| Train Perp: ', trainLoss,
            ', Train Acc: ', trainAcc)
      print('| Valid Perp: ', validLoss,
            ', Valid Acc: ', validAcc)

      if np.isnan(validLoss) or np.isnan(trainLoss):
         print('Got a bad update. Resetting epoch')
         saver.refresh(net)
      else:
         saver.update(net, trainLoss, trainAcc, validLoss, validAcc)

def modelDef(net, cuda=True):
   if cuda: net.cuda()
   utils.initWeights(net)
   utils.modelSize(net)
   return net

def run(cell, depth, h, vocabDim, batchSz, embedDim, embedDrop, 
      context, minContext, eta, saveName, load):
   trainBatcher, validBatcher, testBatcher = dataBatcher(
         batchSz, context, minContext)

   net = modelDef(Network(cell, vocabDim, embedDim, h, 
         vocabDim, context, embedDrop))
   opt = t.optim.Adam(net.parameters(), lr=eta)

   saver = utils.SaveManager(saveName)
   if load: saver.load(net)

   train(net, opt, trainBatcher, validBatcher, saver, minContext)

