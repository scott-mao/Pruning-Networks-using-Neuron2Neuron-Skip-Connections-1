import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.optim as optim
from adj_matrix import adj_matrix
from models.MLP import MLP,MLP_new,MLPexpander 
from LRA import low_rank
from models.layers.skipconnectionlayers import SkipLinear
#from datasets.load_data import LoadCIFAR10
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
#from numpy.linalg import multi_dot
#from collections import Counter
import copy
from train import train,train1,validation,validation1,plots
#import logging
import matplotlib.pyplot as plt
##Getting the models

parser = argparse.ArgumentParser(description='Pytorch Training')
parser.add_argument('--arch',default=MLP,help='Architecture to use')
args = parser.parse_args()


model = MLP() 
print(model)
modelx_6 = MLPexpander(sparsity = 6)
print(modelx_6)

#modelx_15.double() 
#expander_train_loss = train1(modelx_15)
#accuracy = validation1(modelx_15)

'''
W_binary = adj_matrix(model)
Wprune = adj_expander1(modelx)
Wprune_15 = adj_expander1(modelx_15)

D = np.diag(np.sum(np.array(W_binary), axis=1))
Dprune = np.diag(np.sum(np.array(Wprune), axis=1))

L = D - W_binary
Lprune = Dprune - Wprune
'''
X_2 = np.load('logs/X_2_percent.npy')
X_3 = np.load('logs/X_3_percent.npy')
X_4 = np.load('logs/X_4_percent.npy')
X_2_sparse = np.load('logs/X_2_sparse.npy')
X_1_sparse = np.load('logs/X_1_sparse.npy')
'''
fig,ax = plt.subplots()
im = ax.imshow(X)
plt.title("X")
plt.xlim(3000,4042)
plt.ylim(4042,3000)
plt.show()
'''


#Training and testing the models

counter= 0
skip_mean_3 = skip_mean_4 = skip_mean_2 = 0
normal_mean = 0
model_skip_1_sparse_mean = model_skip_2_sparse_mean = expander_mean_4 = expander_mean_6 = expander_mean_7 = expander_mean_8 = 0
maximum_model_skip_1_sparse = maximum_model_skip_2_sparse = maximum_skip_3 = maximum_skip_4 = maximum_skip_2 = maximum_expander_4 = maximum_expander_6 = maximum_expander_7 = maximum_expander_8 = maximum_normal = -1000
minimum_model_skip_1_sparse = minimum_model_skip_2_sparse = minimum_skip_3 = minimum_skip_4 = minimum_skip_2 = minimum_expander_4 = minimum_expander_6 = minimum_expander_7 = minimum_expander_8 = minimum_normal = 1000
while(counter<3): 
  model = MLP().cuda()
  model_skip_3 = MLP_new(X_3).cuda().double()
  model_skip_4 = MLP_new(X_4).cuda().double()
  model_skip_2 = MLP_new(X_2).cuda().double()
  modelx_4 = MLPexpander(sparsity = 4).cuda().double()
  modelx_6 = MLPexpander(sparsity = 6).cuda().double()
  modelx_7 = MLPexpander(sparsity = 7).cuda().double()
  modelx_8 = MLPexpander(sparsity = 8).cuda().double()
  model_skip_2_sparse = MLP_new(X_2_sparse).cuda().double()
  model_skip_1_sparse = MLP_new(X_1_sparse).cuda().double()
  print("******************************Pass: ", counter+1,"*******************************")
  '''
  print("*******************Training the Skip model 1% (total - 2%) ******************")
  skip_training_loss_1_sparse,val_skip_1_sparse = train1(model_skip_1_sparse)
  model_skip_1_sparse_accuracy = validation1(model_skip_2_sparse)

  model_skip_1_sparse_mean += model_skip_1_sparse_accuracy
  if(model_skip_1_sparse_accuracy > maximum_model_skip_1_sparse):
    maximum_model_skip_1_sparse = model_skip_1_sparse_accuracy
  if(model_skip_1_sparse_accuracy < minimum_model_skip_1_sparse):
    minimum_model_skip_1_sparse = model_skip_1_sparse_accuracy
  '''
  
  print("*******************Training the Skip model 2% (total - 4%) ******************")
  skip_training_loss_2_sparse,val_skip_2_sparse = train1(model_skip_2_sparse)
  model_skip_2_sparse_accuracy = validation1(model_skip_2_sparse)
  '''
  print("*******************Training the Expander model - 6% ******************")
  #modelx_15.double()
  expander_training_loss_6,val_expander_6 = train1(modelx_6)
  #   expander_training_loss_mean += (expander_training_loss)
  #   expander_validation_loss_mean += expander_validation_loss

  expander_accuracy_6 = validation1(modelx_6)
  expander_mean_6 += expander_accuracy_6

  if(expander_accuracy_6 > maximum_expander_6):
    maximum_expander_6 = expander_accuracy_6
  if(expander_accuracy_6 < minimum_expander_6):
    minimum_expander_6 = expander_accuracy_6
  
  print("*******************Training the Expander model - 4% ******************")
  #modelx_15.double()
  expander_training_loss_4,val_expander_4 = train1(modelx_4)
  #   expander_training_loss_mean += (expander_training_loss)
  #   expander_validation_loss_mean += expander_validation_loss

  expander_accuracy_4 = validation1(modelx_4)
  expander_mean_4 += expander_accuracy_4

  if(expander_accuracy_4 > maximum_expander_4):
    maximum_expander_4 = expander_accuracy_4
  if(expander_accuracy_4 < minimum_expander_4):
    minimum_expander_4 = expander_accuracy_4

  
  print("*******************Training the Expander model - 7% ******************")
  #modelx_15.double()
  expander_training_loss_7,val_expander_7 = train1(modelx_7)
  #   expander_training_loss_mean += (expander_training_loss)
  #   expander_validation_loss_mean += expander_validation_loss

  expander_accuracy_7 = validation1(modelx_7)
  expander_mean_7 += expander_accuracy_7

  if(expander_accuracy_7 > maximum_expander_7):
    maximum_expander_7 = expander_accuracy_7
  if(expander_accuracy_7 < minimum_expander_7):
    minimum_expander_7 = expander_accuracy_7

  print("*******************Training the Expander model - 8% ******************")
  #modelx_15.double()
  expander_training_loss_8,val_expander_8 = train1(modelx_8)
  #   expander_training_loss_mean += (expander_training_loss)
  #   expander_validation_loss_mean += expander_validation_loss

  expander_accuracy_8 = validation1(modelx_8)
  expander_mean_8 += expander_accuracy_8

  if(expander_accuracy_8 > maximum_expander_8):
    maximum_expander_8 = expander_accuracy_8
  if(expander_accuracy_8 < minimum_expander_8):
    minimum_expander_8 = expander_accuracy_8
  '''
  print("*******************Training the Skip model 3% (total - 6%) ******************")  
  skip_training_loss_3,val_skip_3 = train1(model_skip_3)
  skip_accuracy_3 = validation1(model_skip_3)
  print("*******************Training the Skip model 4% (total - 6%) ******************")
  skip_training_loss_4,val_skip_4 = train1(model_skip_4)
  skip_accuracy_4 = validation1(model_skip_4)
  print("*******************Training the Skip model 2% (total - 6%) ******************")
  skip_training_loss_2,val_skip_2 = train1(model_skip_2)
  skip_accuracy_2 = validation1(model_skip_2)
  
  model_skip_2_sparse_mean += model_skip_2_sparse_accuracy
  if(model_skip_2_sparse_accuracy > maximum_model_skip_2_sparse):
    maximum_model_skip_2_sparse = model_skip_2_sparse_accuracy
  if(model_skip_2_sparse_accuracy < minimum_model_skip_2_sparse):
    minimum_model_skip_2_sparse = model_skip_2_sparse_accuracy


  skip_mean_3 += skip_accuracy_3 
  if(skip_accuracy_3 > maximum_skip_3):
    maximum_skip_3 = skip_accuracy_3
  if(skip_accuracy_3 < minimum_skip_3):
    minimum_skip_3 = skip_accuracy_3

  skip_mean_4 += skip_accuracy_4
  if(skip_accuracy_4 > maximum_skip_4):
    maximum_skip_4 = skip_accuracy_4
  if(skip_accuracy_4 < minimum_skip_4):
    minimum_skip_4 = skip_accuracy_4

  skip_mean_2 += skip_accuracy_2
  if(skip_accuracy_2 > maximum_skip_2):
    maximum_skip_2 = skip_accuracy_2
  if(skip_accuracy_2 < minimum_skip_2):
    minimum_skip_2 = skip_accuracy_2
  '''
  print("*******************Training the Normal model******************")
  normal_training_loss,val_normal = train(model)
  #   expander_training_loss_mean += (expander_training_loss)
  #   expander_validation_loss_mean += expander_validation_loss

  normal_accuracy = validation(model)
  normal_mean += normal_accuracy

  if(normal_accuracy > maximum_normal):
    maximum_normal = normal_accuracy
  if(normal_accuracy < minimum_normal):
    minimum_normal = normal_accuracy
  '''
  counter += 1
  
print("************************** END OF ALL PASSES ***************************")

counter = 3
normal_mean /= counter
expander_mean_6 /= counter
expander_mean_4 /= counter
expander_mean_7 /= counter
expander_mean_8 /= counter
skip_mean_3 /= counter
skip_mean_4 /= counter
skip_mean_2 /= counter
model_skip_2_sparse_mean /= counter
#model_skip_1_sparse_mean /= counter
#print("Skip 1% Accuracy (total 2%) :  ",model_skip_1_sparse_mean, "+-", max(abs(model_skip_1_sparse_mean-minimum_model_skip_1_sparse),abs(model_skip_1_sparse_mean-maximum_model_skip_1_sparse)))
print("Skip 2% Accuracy (total 4%):  ",model_skip_2_sparse_mean, "+-", max(abs(model_skip_2_sparse_mean-minimum_model_skip_2_sparse),abs(model_skip_2_sparse_mean-maximum_model_skip_2_sparse)))
print("Skip 2% Accuracy (total 6%) :  ",skip_mean_2, "+-", max(abs(skip_mean_2-minimum_skip_2),abs(skip_mean_2-maximum_skip_2)))
print("Skip 3% Accuracy (total 6%) :  ",skip_mean_3, "+-", max(abs(skip_mean_3-minimum_skip_3),abs(skip_mean_3-maximum_skip_3)))
print("Skip 4% Accuracy (total 6%) :  ",skip_mean_4, "+-", max(abs(skip_mean_4-minimum_skip_4),abs(skip_mean_4-maximum_skip_4)))
#print("Skip 10% Accuracy:  ",skip_mean_10, "+-", max(abs(skip_mean_10-minimum_skip_10),abs(skip_mean_10-maximum_skip_10)))
#print("Skip 12% Accuracy:  ",skip_mean_12, "+-", max(abs(skip_mean_12-minimum_skip_12),abs(skip_mean_12-maximum_skip_12)))
#print("Expander 6% Final Accuracy:  ",expander_mean_6, "+-",max(abs(expander_mean_6-minimum_expander_6),abs(expander_mean_6-maximum_expander_6)))
#print("Expander 4% Final Accuracy:  ",expander_mean_4, "+-",max(abs(expander_mean_4-minimum_expander_4),abs(expander_mean_4-maximum_expander_4)))
#print("Expander 7% Final Accuracy:  ",expander_mean_7, "+-",max(abs(expander_mean_7-minimum_expander_7),abs(expander_mean_7-maximum_expander_7)))
#print("Expander 8% Final Accuracy:  ",expander_mean_8, "+-",max(abs(expander_mean_8-minimum_expander_8),abs(expander_mean_8-maximum_expander_8)))
#print("Normal Final Accuracy:  ",normal_mean, "+-", max(abs(normal_mean-minimum_normal),abs(normal_mean-maximum_normal)))
# print(normal_mean, "+-", max(abs(normal_mean-minimum_normal),abs(normal_mean-maximum_normal)))
plots(skip_training_loss_2,val_skip_2)
plots(expander_training_loss_8,val_expander_8)
plots(normal_training_loss,val_normal)


