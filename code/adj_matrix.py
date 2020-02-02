from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import numpy as np
from models.layers.expandergraphlayer import ExpanderLinear,ExpanderConv2d

def adj_matrix(model,numclasses):
  dim = 0
  for m in model.modules():
    if isinstance(m,ExpanderLinear) or isinstance(m,nn.Linear):
      dim += m.weight.shape[1]
      #print(dim)
    if isinstance(m,ExpanderConv2d):
      dim += m.fpWeight.shape[1]
      #print(dim)
    elif isinstance(m,nn.Conv2d):
      dim += m.weight.shape[1]
      #print(dim)
  dim += numclasses
  #print(dim)
  adj_matrix = np.zeros((dim,dim))

  k = 0
  kk = 0
  for m in model.modules():
    if isinstance(m,nn.Conv2d):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.weight[:,:,0,0].detach().cpu().numpy()
        kk += m.weight.shape[1]
    if isinstance(m,ExpanderConv2d):
        k += m.fpWeight.shape[1]
#         print(k)
        adj_matrix[k:k+m.fpWeight.shape[0],kk:kk+m.fpWeight.shape[1]] = m.mask[:,:,0,0].detach().cpu().numpy()
        kk += m.fpWeight.shape[1]

    if isinstance(m,nn.Linear):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.weight.detach().cpu().numpy()
        kk += m.weight.shape[1]
    if isinstance(m,ExpanderLinear):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.mask.detach().cpu().numpy()
        kk += m.weight.shape[1]

  low_tri_ind = np.tril_indices(dim, 0) #Get the indices of lower triangle
  adj_matrix.T[low_tri_ind] = adj_matrix[low_tri_ind]
  adj_matrix = np.absolute(adj_matrix)
  adj_matrix = np.where(adj_matrix>0,1,0)
  assert np.allclose(adj_matrix,np.transpose(adj_matrix))
  return adj_matrix

def adj_matrix_weighted(model,numclasses):
  dim = 0
  for m in model.modules():
    if isinstance(m,ExpanderLinear) or isinstance(m,nn.Linear):
      dim += m.weight.shape[1]
      #print(dim)
    if isinstance(m,ExpanderConv2d):
      dim += m.fpWeight.shape[1]
      #print(dim)
    elif isinstance(m,nn.Conv2d):
      dim += m.weight.shape[1]
      #print(dim)
  dim += numclasses
  #print(dim)
  adj_matrix = np.zeros((dim,dim))

  k = 0
  kk = 0
  for m in model.modules():
    if isinstance(m,nn.Conv2d):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.weight[:,:,0,0].detach().cpu().numpy()
        kk += m.weight.shape[1]
    if isinstance(m,ExpanderConv2d):
        k += m.fpWeight.shape[1]
#         print(k)
        adj_matrix[k:k+m.fpWeight.shape[0],kk:kk+m.fpWeight.shape[1]] = m.mask[:,:,0,0].detach().cpu().numpy()
        kk += m.fpWeight.shape[1]

    if isinstance(m,nn.Linear):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.weight.detach().cpu().numpy()
        kk += m.weight.shape[1]
    if isinstance(m,ExpanderLinear):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.mask.detach().cpu().numpy()
        kk += m.weight.shape[1]

  low_tri_ind = np.tril_indices(dim, 0) #Get the indices of lower triangle
  adj_matrix.T[low_tri_ind] = adj_matrix[low_tri_ind]
  adj_matrix = np.absolute(adj_matrix)
  #adj_matrix = np.where(adj_matrix>0,1,0)
  assert np.allclose(adj_matrix,np.transpose(adj_matrix))
  return adj_matrix


def randomsparse(inputs,outputs,sparsity):
  mask = torch.zeros(outputs,inputs)
  # print(mask.shape)
  for i in range(outputs):
      x = torch.randperm(inputs)
      for j in range(int((sparsity/100)*inputs)):
          mask[i][x[j]] = 1
  return mask.cpu().numpy()

def adj_skip(model,skips,sparsity):
  W = adj_matrix(model,10)
  W = np.tril(W)
  k,kk = 0,0
  for m in model.children():
    L = m.weight.shape[1]
    pp,p = kk,0
    if kk < W.shape[1]:
      skip = skips
      for child in model.children():
        if np.count_nonzero(W[p+10:p+20,pp+10:pp+20]) == 0:
          p += child.weight.shape[1]
        else:
          p += child.weight.shape[1]
          W[p:p+child.weight.shape[0],pp:pp+L] = randomsparse(L,child.weight.shape[0],sparsity)
          skip -= 1
          if skip == 0:
            break

    kk += m.weight.shape[1]
  return W

def sparsityfixer(model,skips,sparsity):
  Wskip = adj_skip(model,skips,sparsity)  
  kk,k = 0,0
  for m in model.children():
    breadth = m.weight.shape[1]
    initial_length = m.weight.shape[0]
    if kk < Wskip.shape[1]:
      p,final_length = 0,0
      pp = kk
      skip = skips
      for child in model.children():
        if np.count_nonzero(Wskip[p+10:p+20,pp+10:pp+20]) == 0:
          p += child.weight.shape[1]
          start = p 
        else:
          final_length += child.weight.shape[1]
          skip -= 1
        if skip == 0:
          break

    final_length += child.weight.shape[0]
    end = start+final_length
    Wskip[start:end,kk:kk+breadth] = randomsparse(breadth,final_length,sparsity*initial_length/(final_length))
    kk += breadth 
  return Wskip

def adj_skipconv(model,skips,sparsity):
  W = adj_matrix(model,10)
  W = np.tril(W)
  k,kk = 0,0
  for m in model.modules():
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
      L = m.weight.shape[1]
      pp,p = kk,0
      if kk < W.shape[1]:
        skip = skips
        for child in model.modules():
          if isinstance(child,nn.Conv2d) or isinstance(child,nn.Linear):
            if np.count_nonzero(W[p+10:p+20,pp+10:pp+20]) == 0:
              p += child.weight.shape[1]
            else:
              p += child.weight.shape[1]
              W[p:p+child.weight.shape[0],pp:pp+L] = randomsparse(L,child.weight.shape[0],sparsity)
              skip -= 1
              if skip == 0:
                break
          else:
            continue
      kk += m.weight.shape[1]
    else:
      continue
  return W

def sparsityfixerconv(model,skips,sparsity):
  Wskip = adj_skipconv(model,skips,sparsity)  
  kk,k = 0,0
  for m in model.modules():
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
      breadth = m.weight.shape[1]
      initial_length = m.weight.shape[0]
      if kk < Wskip.shape[1]:
        p,final_length = 0,0
        pp = kk
        skip = skips
        for child in model.modules():
          if isinstance(child,nn.Conv2d) or isinstance(child,nn.Linear):
            if np.count_nonzero(Wskip[p+10:p+20,pp+10:pp+20]) == 0:
              p += child.weight.shape[1]
              start = p 
            else:
              final_length += child.weight.shape[1]
              skip -= 1
            if skip == 0:
              break
          else:
            continue
    else:
      continue
    final_length += child.weight.shape[0]
    end = start+final_length
    Wskip[start:end,kk:kk+breadth] = randomsparse(breadth,final_length,sparsity*initial_length/(final_length))
    kk += breadth 
  return Wskip
