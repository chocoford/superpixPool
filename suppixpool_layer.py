import torch
import suppixpool_CUDA as spx_gpu
import numpy as np

class AvgSupPixPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, spx):
        spx = spx.to(torch.int)
        K = spx.max()+1
        assert(spx.size()[-2:]==img.size()[-2:])
        out = spx_gpu.avg_forward(img, spx, K)
        outputs, pool_size = out
        outputs /= pool_size.to(torch.float)
        ctx.save_for_backward(pool_size, img, spx, K)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        pool_size, img, spx, K = ctx.saved_tensors
        grad_input = grad_output / pool_size.to(torch.float)
        grad_input = SupPixUnpool()(grad_input, spx.long())
        return grad_input, torch.zeros_like(spx)

class AvgSupPixPool(torch.nn.Module):
    def __init__(self):
        super(AvgSupPixPool, self).__init__()

    def forward(self, img, spx):
        return AvgSupPixPoolFunction.apply(img, spx)

class MaxSupPixPoolFunction(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, img, spx):
        spx = spx.to(torch.int)   
        K = spx.max()+1
        assert(spx.size()[-2:]==img.size()[-2:])
        # print(np.all(np.arange(K)==np.unique(spx.cpu().numpy())))
        # print "used K: ", K
        out = spx_gpu.max_forward(img, spx, K)
        outputs, indices = out
        # print("(max, min) indices: ", indices.max(), indices.min())
        # print("number of -1: ", indices.eq(-1).sum())
        # print indices
        # assert np.all(indices.cpu().numpy()>=0)

        ctx.save_for_backward(indices, img, spx, K)
        return outputs
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        indices, img, spx, K = ctx.saved_tensors
        grad_input, = spx_gpu.max_backward(grad_output.contiguous(), img, spx, indices, K)
        return grad_input, torch.zeros_like(spx)

class MaxSupPixPool(torch.nn.Module):
    def __init__(self):
        super(MaxSupPixPool, self).__init__()

    def forward(self, img, spx):
        return MaxSupPixPoolFunction.apply(img, spx)

class SupPixUnpool(torch.nn.Module):
    def __init__(self):
        super(SupPixUnpool, self).__init__()

    def forward(self, pooled, spx):
        outShape = pooled.size()[0:2]+spx.size()[-2:]
        out = pooled.new_zeros(outShape)
        for batch in range(pooled.size()[0]):
            out[batch, :, :, :] = pooled[batch, :, spx[batch,:,:]]
        return out
