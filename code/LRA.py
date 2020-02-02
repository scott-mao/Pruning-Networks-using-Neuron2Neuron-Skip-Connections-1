import numpy as np
from numpy.linalg import multi_dot
 
def low_rank(W,valuesW,vectorsW):
  from numpy.linalg import multi_dot
  A_W = np.copy(W)
  lamdaW_sort = np.diag(valuesW)
  raw = multi_dot([vectorsW,lamdaW_sort,np.transpose(vectorsW)])
  raw = np.absolute(raw)
  return raw
