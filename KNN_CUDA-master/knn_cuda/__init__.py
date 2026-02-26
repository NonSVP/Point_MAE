import os
import torch
import torch.nn as nn

# Version of the wrapper
__version__ = "0.2"

# Instead of JIT loading (which fails on Windows), 
# we import the .pyd file we just built.
try:
    from . import _ext as _knn
except ImportError:
    raise ImportError(
        "Could not import the compiled _ext library. "
        "Make sure you ran 'python setup.py build_ext --inplace' successfully "
        "and that the .pyd file exists in the knn_cuda folder."
    )

def knn(ref, query, k):
    # Our compiled function is inside _ext
    d, i = _knn.knn(ref, query, k)
    # The original code subtracts 1 because the CUDA kernel 
    # uses 1-based indexing for some reason.
    i -= 1
    return d, i

def _T(t, mode=False):
    if mode:
        return t.transpose(0, 1).contiguous()
    else:
        return t

class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                r, q = _T(ref[bi], self._t), _T(query[bi], self._t)
                d, i = knn(r.float(), q.float(), self.k)
                d, i = _T(d, self._t), _T(i, self._t)
                D.append(d)
                I.append(i)
            D = torch.stack(D, dim=0)
            I = torch.stack(I, dim=0)
        return D, I