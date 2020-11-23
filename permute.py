#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-11-23 15:23:00 (UTC+0100)

import sys
import numpy
import scipy.optimize as optimize
from optimap import optimap
import torch


def permoptim(A, B, P=None):
    """
    Find a permutation P that minimizes the sum of square errors ||AP-B||^2
    See: https://math.stackexchange.com/a/3226657/192193
    >>> A = numpy.random.uniform(size=(3,3))
    >>> P0 = numpy.asarray([[1,0,0], [0,0,1], [0,1,0]])
    >>> B = A.dot(P0)
    >>> P = permoptim(A, B)
    >>> (P == P0).all()
    True
    >>> (A.dot(P) == B).all()
    True

    >>> A = numpy.random.uniform(size=(4,4))
    >>> P0 = numpy.asarray([[1,0,0], [0,0,1], [0,1,0], [0, 0, 0]])
    >>> B = A.dot(P0)
    >>> B = B[:3]
    >>> B.shape
    (3, 3)
    >>> P = permoptim(A, B)
    >>> (P == P0).all()
    True
    >>> (A.dot(P)[:B.shape[0]] == B).all()
    True
    """
    n, p = A.shape[0], B.shape[0]
    B_expand = numpy.zeros((A.shape[1], B.shape[0]))
    B_expand[:p] = B
    if P is None:
        P = numpy.eye(A.shape[0])
    C = A.T.dot(P.T).dot(B_expand)
    costmat = C.max() - C
    costmat[p:] = 9999.99
    costmat[:, p:] = 9999.99
    row_ind, col_ind = optimize.linear_sum_assignment(costmat)
    P = numpy.zeros((n, n))
    assignment = -numpy.ones(n, dtype=int)
    assignment[col_ind] = row_ind
    assignment = assignment[assignment > -1]
    P[assignment, numpy.arange(len(assignment))] = 1.
    P = P.T
    P = P[:, P.sum(axis=0) != 0]
    return P


def permiter(coords, coords_ref, n_step=100):
    A = optimap.get_cmap(torch.from_numpy(coords1), 'cpu', ca_switch=True).numpy()
    B = optimap.get_cmap(torch.from_numpy(coords_ref), 'cpu', ca_switch=True).numpy()
    P = permoptim(A, B)
    coords_P = P.dot(coords)
    A_optim = optimap.get_cmap(torch.from_numpy(coords_P), 'cpu', ca_switch=True).numpy()
    for i in range(n_step):
        P = permoptim(A_optim, B, P)
        coords_P = P.dot(coords_P)
        A_optim = optimap.get_cmap(torch.from_numpy(coords_P), 'cpu', ca_switch=True).numpy()
        score = ((A_optim - B)**2).sum()
        sys.stdout.write(f'{i+1}/{n_step} {score}        \r')
        sys.stdout.flush()
    return A_optim


if __name__ == '__main__':
    import argparse
    import doctest
    import matplotlib.pyplot as plt
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', default=False, action='store_true', help='test the code')
    parser.add_argument('--coords1', help='Filename of coordinates to permute', type=str)
    parser.add_argument('--coords2', help='Filename of reference coordinates', type=str)
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit(0)
    
    coords1 = numpy.genfromtxt(args.coords1)
    coords_ref = numpy.genfromtxt(args.coords2)
    A = optimap.get_cmap(torch.from_numpy(coords1), 'cpu', ca_switch=True).numpy()
    B = optimap.get_cmap(torch.from_numpy(coords_ref), 'cpu', ca_switch=True).numpy()
    plt.matshow(A)
    plt.savefig('cmap_shuf.png')
    plt.matshow(B)
    plt.savefig('cmap_ref.png')
    P = permoptim(A, B)
    coords_P = P.dot(coords1)
    A_P = optimap.get_cmap(torch.from_numpy(coords_P), 'cpu', ca_switch=True).numpy()
    # plt.matshow(A_P)
    # plt.savefig('cmap_P.png')
    A_optim = permiter(coords1, coords_ref, n_step=1000)
    plt.matshow(A_optim)
    plt.savefig('cmap_optim.png')
