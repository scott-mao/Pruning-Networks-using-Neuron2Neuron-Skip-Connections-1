import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from adj_matrix import adj_matrix,adj_expander1
from LRA import low_rank
from models.layers.skipconnectionlayers import SkipLinear,NewLinear
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from models.MLP_lenet import MLP_lenet,MLPexpander_lenet,MLP_new_lenet

##Normal Model
model_lenet = MLP_lenet()
W_binary_lenet = adj_matrix(model_lenet)
#valuesW_lenet,vectorsW_lenet = np.linalg.eig(W_binary_lenet)

##20% Expander Model
print("*******************20% expander****************")
modelx_20_lenet = MLPexpander_lenet(sparsity = 20).cuda()
Wprune_20_lenet = adj_expander1(modelx_20_lenet)
Dprune_20_lenet = np.diag(np.sum(np.array(Wprune_20_lenet),axis = 1))
Lprune_20_lenet = Dprune_20_lenet - Wprune_20_lenet
eprune_20_lenet,vprune_20_lenet = np.linalg.eig(Wprune_20_lenet)

raw, recon = low_rank(Wprune_20_lenet,600,'d',eprune_20_lenet,vprune_20_lenet)

def skip_matrix(raw,percent,y):
  model_lenet = MLP_lenet()
  W_binary_lenet = adj_matrix(model_lenet)
  modelx_2_lenet = MLPexpander_lenet(sparsity=2)
  modelx_20_lenet = MLPexpander_lenet(sparsity=20)
  
  Wprune_2_lenet = adj_expander1(modelx_2_lenet)
  Wprune_20_lenet = adj_expander1(modelx_20_lenet)

  Matrix = np.ones((Wprune_20_lenet.shape[0],Wprune_20_lenet.shape[1]))
  k=0
  kk=0
  for m in model_lenet.children():
    Matrix[k:k+m.weight.shape[1],kk:kk+m.weight.shape[1]] = 0
    kk += m.weight.shape[1]
    k += m.weight.shape[1]

  intra_layer = np.real(np.multiply(raw,Matrix))

  diagonal = np.multiply(intra_layer,W_binary_lenet)
  skip = intra_layer - diagonal

  print("Range of skip connections:",np.min(skip),np.max(skip))
  print("Range of Normal connections:",np.min(diagonal),np.max(diagonal))

  ###Layer-wise
  skip_binary = np.zeros_like(skip)

  k = 0
  kk = 0
  for m in model_lenet.children():
    k += m.weight.shape[1]
    if(m.weight.shape[0] == 10):
      break

    #### Getting the skip layer
    x = percent*np.count_nonzero(W_binary_lenet[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]])/np.count_nonzero(skip[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]])
    threshold = 100-x
    thre = np.percentile(skip[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]],threshold)
    skip_binary[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]] = np.where(skip[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]]>thre,1,0)
    kk += m.weight.shape[1]

  low_tri_ind = np.tril_indices(skip_binary.shape[0],0)
  skip_binary.T[low_tri_ind] = skip_binary[low_tri_ind]

  X = skip_binary + adj_expander1(MLPexpander_lenet(y-percent))
  return skip_binary,X

skip_binary_3_lenet,X_3_lenet = skip_matrix(raw,1.5,3)
skip_binary_2_lenet,X_2_lenet = skip_matrix(raw,2,4)
skip_binary_1_lenet,X_1_lenet = skip_matrix(raw,1,2)
skip_binary_100_lenet,X_100_lenet = skip_matrix(raw,0.5,1)
'''
for m in modelx_20_lenet.children():
  print(100*np.count_nonzero(m.mask.data.cpu().numpy())/np.count_nonzero(m.weight.data.cpu().numpy()))
'''
np.save('logs/X_1_lenet.npy',X_1_lenet)
np.save('logs/X_2_lenet.npy',X_2_lenet)
np.save('logs/X_100_lenet.npy',X_100_lenet)
np.save('models/layers/X_lenet.npy',X_1_lenet)
np.save('logs/X_3_lenet.npy',X_3_lenet)

print(np.count_nonzero(X_100_lenet)/np.count_nonzero(W_binary_lenet))
