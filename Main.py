from pdb import set_trace as T
from Run import run
from models.BaselineLSTM import LSTMCell
from models.RHN import RHNCell
from models.HyperRHN import HyperRHNCell

saveName = 'saves/test/'
load = False
cellFunc = HyperRHNCell
depth = 2
h = 128
hCell = h
hCell = (128, h)
vocab = 50
embedDim = 27
batchSz = 100
context = 100
minContext = 50
eta = 1e-3
gateDrop = embedDrop = 0.65

cell = cellFunc(embedDim, hCell, depth, gateDrop)
run(cell, depth, h, vocab, batchSz, embedDim, embedDrop,
      context, minContext, eta, saveName, load)
