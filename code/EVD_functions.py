import numpy as np
import torch
import torchvision
import torch.nn as nn
from adj_matrix import adj_matrix, adj_matrix_weighted
from models.layers.expandergraphlayer import ExpanderLinear, ExpanderConv2d
from train import validation_conv
import time
#from Random import RandomLinear,RandomConv2d
#from Skip import SkipLinear,SkipConv2d
import os
from LRA import low_rank
import scipy

numclasses = 10

def EVD(model,rank):
  #model = MLP()
  W_binary = adj_matrix(model,10)
  W_raw = adj_matrix_weighted(model,10) 
  #Wprune = adj_matrix(modelx,10)
  #print("The size of the expander 20 adjacency matrix is ",Wprune_20.shape, W_binary.shape)   
  #Eigen Value Decomposition
  D = np.diag(np.sum(np.tril(W_binary), axis=1))
  #Dprune = np.diag(np.sum(np.array(Wprune), axis=1))

  L = D - W_binary

  '''20% expander'''
  import scipy
  from scipy.sparse.linalg import eigs

  #Low rank approximation of the Baseline Model
  since2 = time.time()
  #Wprune = scipy.sparse.csr_matrix(Wprune).astype(float)  
  #eprune,vprune = eigs(Wprune,k=600,which='LR')
  W_binary = scipy.sparse.csr_matrix(W_binary).astype(float)
  e,v = eigs(W_binary,k=rank,which='LR')
  # print(e2)
  print("Time Elapsed :", time.time()-since2,"seconds")

  #Wprune = adj_matrix(modelx,10)
  W_binary = adj_matrix(model,10)
  return W_binary,e,v


def skip_matrix(raw,percent,y,MLPexpander,model):
  W_binary = adj_matrix(model,10)
  Matrix = np.ones((W_binary.shape[0],W_binary.shape[1]))
  k=0
  kk=0
  for m in model.modules():
    if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d) or isinstance(m,ExpanderLinear):
      Matrix[k:k+m.weight.shape[1],kk:kk+m.weight.shape[1]] = 0
      kk += m.weight.shape[1]
      k += m.weight.shape[1]
    if isinstance(m,ExpanderConv2d):
      Matrix[k:k+m.fpWeight.shape[1],kk:kk+m.fpWeight.shape[1]] = 0
      kk += m.fpWeight.shape[1]
      k += m.fpWeight.shape[1]

  intra_layer = np.real(np.multiply(raw,Matrix)) ##Use this for the weighted adjacency matrix


  diagonal = np.multiply(intra_layer,W_binary)
  skip = intra_layer - diagonal

  ###Layer-wise
  skip_binary = np.zeros_like(skip)

  k = 0
  kk = 0
  for m in model.modules():
    if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d) or isinstance(m,ExpanderLinear):
      k += m.weight.shape[1]
      if(m.weight.shape[0] == numclasses):
        break
    #### Getting the skip layer
      x = percent*np.count_nonzero(W_binary[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]])/np.count_nonzero(skip[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]])
      if x>=100:
        x = 100
      #print(x) 
      threshold = 100-x
      thre = np.percentile(skip[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]],threshold)
      skip_binary[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]] = np.where(skip[k+m.weight.shape[0]:,kk:kk+m.weight.shape[1]]>thre,1,0)
      kk += m.weight.shape[1]

    if isinstance(m,ExpanderConv2d):
      k += m.fpWeight.shape[1]
      if(m.fpWeight.shape[0] == numclasses):
        break
      x = percent*np.count_nonzero(W_binary[k:k+m.fpWeight.shape[0],kk:kk+m.fpWeight.shape[1]])/np.count_nonzero(skip[k+m.fpWeight.shape[0]:,kk:kk+m.fpWeight.shape[1]])
      if x>=100:
        x = 100
      threshold = 100-x
      thre = np.percentile(skip[k+m.fpWeight.shape[0]:,kk:kk+m.fpWeight.shape[1]],threshold)
      skip_binary[k+m.fpWeight.shape[0]:,kk:kk+m.fpWeight.shape[1]] = np.where(skip[k+m.fpWeight.shape[0]:,kk:kk+m.fpWeight.shape[1]]>thre,1,0)
      kk += m.weight.shape[1]

  low_tri_ind = np.tril_indices(skip_binary.shape[0],0)
  skip_binary.T[low_tri_ind] = skip_binary[low_tri_ind]


  #expandcfg = vgg_sparsity(y-percent,cfg) 

  X = skip_binary + adj_matrix(MLPexpander(y-percent),10)
  return skip_binary,X

