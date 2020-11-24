#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-11-23 15:23:00 (UTC+0100)

import sys
import numpy
import scipy.optimize as optimize
import scipy.spatial.distance as distance
from pymol import cmd


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def get_coords(pdbfilename, object, selection=None):
    if selection is None:
        selection = f'{object} and name CA'
    else:
        selection = f'{object} and name CA and {selection}'
    cmd.load(pdbfilename, object=object)
    cmd.remove(f'not ({selection}) and {object}')
    coords = cmd.get_coords(selection=object)
    return coords


def get_cmap(coords, threshold=8., ca_switch=False, dist_ca=3.8, sigma_ca=0.1):
    """
    - ca_switch: if True, apply a different distance threshold for consecutive CA
    - dist_ca: C-alpha - C-alpha distance
    """
    pdist = distance.squareform(distance.pdist(coords))
    cmap_S = sigmoid(threshold - pdist)
    if ca_switch:
        n = coords.shape[0]
        A = numpy.meshgrid(numpy.arange(n), numpy.arange(n))
        dist_to_diag = numpy.abs(A[1] - A[0])
        cmap_G = numpy.exp(-(pdist - dist_ca)**2 / (2 * sigma_ca**2))
        mask = dist_to_diag == 1
        cmap = numpy.where(mask, cmap_G, cmap_S)
    else:
        cmap = cmap_S
    return cmap


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


def permiter(coords, cmap_ref, n_step=100):
    A = get_cmap(coords1, ca_switch=True)
    B = cmap_ref
    P = permoptim(A, B)
    P_total = P.copy()
    coords_P = P.dot(coords)
    A_optim = get_cmap(coords_P, ca_switch=True)
    with open('permiter.log', 'w') as logfile:
        for i in range(n_step):
            P = permoptim(A_optim, B, P)
            P_total = P.dot(P_total)
            coords_P = P.dot(coords_P)
            A_optim = get_cmap(coords_P, ca_switch=True)
            score = ((A_optim - B)**2).sum()
            logfile.write(f'{i+1}/{n_step} {score}\n')
            sys.stdout.write(f'{i+1}/{n_step} {score}        \r')
            sys.stdout.flush()
    print()
    return get_cmap(P_total.dot(coords), ca_switch=True)


if __name__ == '__main__':
    import argparse
    import doctest
    import matplotlib.pyplot as plt
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', default=False, action='store_true', help='test the code')
    parser.add_argument('--pdb1', help='pdb filename of coordinates to permute', type=str)
    parser.add_argument('--pdb2', help='pdb filename of reference coordinates', type=str)
    parser.add_argument('--cmap', help='npy file of the target -- reference -- contact map', type=str)
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit(0)
    
    coords1 = get_coords(args.pdb1, 'shuf')
    A = get_cmap(coords1, ca_switch=True)
    if args.pdb2 is not None:
        coords_ref = get_coords(args.pdb2, 'ref')
        B = get_cmap(coords_ref, ca_switch=True)
    else:
        B = numpy.load(args.cmap)
    plt.matshow(A)
    plt.savefig('cmap_shuf.png')
    plt.matshow(B)
    plt.savefig('cmap_ref.png')
    P = permoptim(A, B)
    coords_P = P.dot(coords1)
    A_P = get_cmap(coords_P, ca_switch=True)
    # plt.matshow(A_P)
    # plt.savefig('cmap_P.png')
    A_optim = permiter(coords1, B, n_step=1000)
    plt.matshow(A_optim)
    plt.savefig('cmap_optim.png')
