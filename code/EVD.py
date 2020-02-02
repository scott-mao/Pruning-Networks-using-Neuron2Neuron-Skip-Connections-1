import time
import torch
import torch.nn as nn
from adj_matrix import adj_matrix
from models.MLP import MLP,MLP_new,MLPexpander
from models.conv import Net,Netexpander
from LRA import low_rank
from models.layers.skipconnectionlayers import SkipLinear
import numpy as np
from numpy.linalg import multi_dot
from models.layers.expandergraphlayer import ExpanderLinear,ExpanderConv2d
#from models.VGG import VGG, vgg11,vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn,vgg19_bn, vgg19
from EVD_functions import EVD, skip_matrix

MLP = MLP()
model = Net()
modelx = Netexpander(sparsity=20).cuda()
MLPx = MLPexpander(sparsity=20).cuda()
#print("Getting Expander Model")
W_binary = adj_matrix(model,10)
WMLP = adj_matrix(MLP,10)
#valuesW,vectorsW = np.linalg.eig(W_binary)
Wprune_20 = adj_matrix(modelx,10)
print(Wprune_20.shape)
#vgg16 = vgg16()
#W_vgg16 = adj_matrix(vgg16,10)


since = time.time()
Wprune,eprune,vprune = EVD(modelx)
Wprunemlp,emlp,vmlp = EVD(MLPx)

raw = low_rank(Wprune,eprune,vprune)
raw1 = low_rank(Wprunemlp,emlp,vmlp)
print("Time Elapsed: ", time.time()-since)

skip_binary_1_2,X_2_sparse = skip_matrix(raw1,2,4,MLPexpander,MLP)
skip_2,X_2_percent = skip_matrix(raw1,2,6,MLPexpander,MLP)
skip_3,X_3_percent = skip_matrix(raw1,3,6,MLPexpander,MLP)
skip_4,X_4_percent = skip_matrix(raw1,4,6,MLPexpander,MLP)
skip_binary_4, X_4 = skip_matrix(raw,10,20,Netexpander,model)


print("Total Connections - ",100*np.count_nonzero(X_2_percent)/np.count_nonzero(WMLP),"%")
print("Total Connections - ",100*np.count_nonzero(skip_2)/np.count_nonzero(WMLP),"%")
print("Total Connections - ",100*np.count_nonzero(X_3_percent)/np.count_nonzero(WMLP),"%")
print("Total Connections - ",100*np.count_nonzero(skip_3)/np.count_nonzero(WMLP),"%")
print("Total Connections - ",100*np.count_nonzero(X_4_percent)/np.count_nonzero(WMLP),"%")
print("Total Connections - ",100*np.count_nonzero(skip_4)/np.count_nonzero(WMLP),"%")


print("Total Connections - ",100*np.count_nonzero(X_4)/np.count_nonzero(W_binary),"%")
print("Only Skip Connections - ",100*np.count_nonzero(skip_binary_4)/np.count_nonzero(W_binary),"%")
np.save('logs/X_4_10.npy',X_4)
np.save('X_4_10.npy',X_4)
np.save('models/X_4_10.npy',X_4)


#np.save('models/layers/X_4_10.npy',X_4)

np.save('logs/X_2_percent.npy',X_2_percent)
np.save('logs/X_3_percent.npy',X_3_percent)
np.save('logs/X_4_percent.npy',X_4_percent)
np.save('logs/X_2_sparse.npy',X_2_sparse)
#np.save('logs/X_1_sparse.npy',X_1_sparse)

