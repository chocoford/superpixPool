from __future__ import print_function
import torch
from suppixpool_layer import MaxSupPixPool, AvgSupPixPool, SupPixUnpool
import torch.nn as nn
import numpy as np
import time
from skimage.segmentation import slic
from torch.autograd import Variable


if __name__ == "__main__":

    GPU = torch.device("cuda:0")
    batch_size = 1
    n_channels = 4
    xSize = 4
    ySize = 4

    X = torch.randn((batch_size,n_channels,xSize,ySize), dtype=torch.float32, device=GPU,requires_grad=True)
    spx = ((np.array([np.arange(xSize*ySize).reshape(xSize,ySize)]*batch_size) / 2) + 0.5).astype(int)
    # spx.dtype = "int64"
    # spx = np.zeros((batch_size, xSize, ySize))
    spx = torch.from_numpy(spx) 
    spx = spx.to(GPU)
    # X + X 
    print ("INPUT ARRAY  ----------------- \n", X) 
    pool = AvgSupPixPool()
    pld = pool(X, spx)
    A = pool.get_adjacent_matrix(spx)
    print(spx)
    print ("adjacent matrix ----------------- ")
    print(A)

    print ("POOLED ARRAY ----------------- \n", pld)
    print ("Shape of pooled array: ", pld.size())
    unpool = SupPixUnpool()
    unpld = unpool(pld, spx)
    # print ("Unpooling back to original: ", np.all(unpld == X))

    res = torch.autograd.gradcheck(pool, (X.double(), spx), raise_exception=False)
    resUnpool = torch.autograd.gradcheck(unpool, (pld.double(), spx), raise_exception=False) 

    print ("Gradients of pooling are {}.".format("correct" if res else "wrong")) # res should be True if the gradients are correct.
    print ("Gradients of unpooling are {}.".format("correct" if resUnpool else "wrong"))

    # avgPool = AvgSupPixPool()
    # avgPld = avgPool(X, spx)