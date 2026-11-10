import math

import torch
import torch.nn.functional as F


def dex_function(x, E, W1, W2):
    '''
    Modulates latent embeddings with a convex combination of Distortion Extenders

    Arg(s):
        x : torch.Tensor[float32]
            B x N x D latent embeddings (tokens or flattened feature vectors)
        E : torch.Tensor[float32]
            M x D Extender modulation parameters
        W1 : torch.Tensor[float32]
            D x d low-rank factor of the selection matrix
        W2 : torch.Tensor[float32]
            d x D low-rank factor of the selection matrix
    Returns:
        torch.Tensor[float32] : B x N x D modulated embeddings
    '''

    D = x.shape[-1]
    W = W1 @ W2

    # Extender selections S = E W, with gradients flowing through W only
    S = E.detach() @ W
    A = F.softmax((x @ S.t()) / math.sqrt(D), dim=-1)

    return x + A @ E
