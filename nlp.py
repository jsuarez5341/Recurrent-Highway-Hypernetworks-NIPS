import numpy as np
import utils

#Generic text vocab generator
def buildVocab(fName):
   dat = open(fName).read()
   chars = sorted(np.unique(list(dat)).tolist())
   vocab = dict(zip(list(chars), np.arange(len(chars))))
   invVocab = {v: k for k, v in vocab.items()}
   return vocab, invVocab

#Generic txt-vocab
def applyVocab(line, vocab):
   ret = []
   for e in line:
      ret += [vocab[e]]
   return np.asarray(ret)

def applyInvVocab(x, vocab):
   x = applyVocab(x, utils.invertDict(vocab))
   return ''.join(x)


