#!/usr/bin/env python

import numpy as np
"""
torch tests
"""

def test_batch_quadratic():
    """
    test the batch mahalanobis against numpy
    """
    from scipy.spatial.distance import mahalanobis
    torch_v = torch.randn(10,3)
    torch_M = 4.*torch.ones(10, 3,3)
    batch_out = batch_quadratic(torch_v, torch_M)

    np_outs = []
    for v, M in zip(torch_v, torch_M):
        np_v, np_M = v.numpy(), M.numpy()
        np_outs.append(mahalanobis(np_v, np.zeros(len(np_v)), np_M)**2)
    assert np.allclose(np.array(np_outs), batch_out.numpy()), f"failed: {np_outs}, {batch_out}"

def test_batch_dot():
    """
    test the batch dot against numpy
    """
    from scipy.spatial.distance import mahalanobis
    from torch_utils import batch_dot
    torch_v = torch.randn(10,3)
    torch_u = torch.randn(10, 3)
    batch_out = batch_dot(torch_v, torch_u)

    np_outs = []
    for v, u in zip(torch_v, torch_u):
        np_v, np_u = v.numpy(), u.numpy()
        np_outs.append(np_v.dot(np_u))
    assert np.allclose(np.array(np_outs), batch_out.numpy()), f"failed: {np_outs}, {batch_out}"
