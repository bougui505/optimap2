#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-11-23 15:23:00 (UTC+0100)

import sys
import os
import numpy
import scipy.optimize as optimize
import scipy.spatial.distance as distance
from scipy.linalg import block_diag
from pymol import cmd
import Traj


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


def get_cmap(coords, threshold=8., ca_switch=True, dist_ca=3.8, sigma_ca=0.1):
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
    """
    n, p = A.shape[0], B.shape[0]
    if P is None:
        if n > p:
            P = numpy.block([numpy.identity(p), numpy.zeros((p, n - p))])
        if n < p:
            P = numpy.block([numpy.identity(n), numpy.zeros((n, p - n))]).T
        if n == p:
            P = numpy.identity(n)
    C = A.T.dot(P.T).dot(B)
    cmax = C.max()
    costmat = cmax - C
    row_ind, col_ind = optimize.linear_sum_assignment(costmat)
    P = numpy.zeros((p, n))
    assignment = -numpy.ones(max(n, p), dtype=int)
    assignment[col_ind] = row_ind
    assignment = assignment[assignment > -1]
    P[numpy.arange(len(assignment)), assignment] = 1.
    # P = P.T
    # P = P[P.sum(axis=0) != 0]
    return P


def add_coords(coords, r):
    """
    Add r zero coordinates to coords
    """
    print(f"Adding {r} coordinates")
    return numpy.block([[coords], [numpy.zeros((r, 3))]])


def zero_mask(coords):
    """
    Mask for the added coordinates
    """
    mask = (coords.sum(axis=1) == 0.)
    return mask


def permute_coords(coords, P, same=True):
    """
    Permute the coordinates using P
    If same is True, return a coordinates array with the same shape as the input coords
    """
    p, n = P.shape
    if p < n and same:  # Works
        sel = (P.sum(axis=0) == 0)
        inds = numpy.where(sel)  # Coordinates not used
        P1 = numpy.zeros((n - p, n))
        P1[range(n - p), inds] = 1
        P = numpy.block([[P], [P1]])
    coords_P = P.dot(coords)
    if same:
        return coords_P, P
    else:
        return coords_P


def permiter(coords, cmap_ref, n_step=10000, save_traj=False, topology=None, outtrajfilename='permiter.dcd'):
    A = get_cmap(coords)
    B = cmap_ref
    n = coords.shape[0]
    p = B.shape[0]
    P = permoptim(A, B)
    coords_P, P = permute_coords(coords, P)
    P_total = P.copy()
    A_optim = get_cmap(coords_P)
    scores = []
    score_steps = []
    if save_traj:
        traj = Traj.Traj(topology)
    with open('permiter.log', 'w') as logfile:
        for i in range(n_step):
            P = permoptim(A_optim[:n, :n], B, P[:p, :n])
            coords_P, P = permute_coords(coords_P[:n, :], P)
            P_total = P.dot(P_total)
            if save_traj:
                traj.append(coords_P)
            A_optim = get_cmap(coords_P)
            mask = zero_mask(coords_P)
            score = ((A_optim[:p][:, :p] - B)**2)[~mask[:p]].sum()
            logfile.write(f'{i+1}/{n_step} {score}\n')
            sys.stdout.write(f'{i+1}/{n_step} {score}                          \r')
            sys.stdout.flush()
            scores.append(score)
            # if (numpy.isclose(scores[-1], scores, atol=1e-3, rtol=0)).sum() > 10:
            if (scores[-1] == numpy.asarray(scores)).sum() > 3:
                print()
                score_steps.append(scores[-1])
                _, counts = numpy.unique(score_steps, return_counts=True)
                count = max(counts)
                if count > 3:
                    print("Early stop")
                    break
                else:
                    print("Adding noise to escape local minima")
                    noise = numpy.random.uniform(size=P.shape)
                    P += noise
    print()
    if save_traj:
        traj.save(outtrajfilename)
    return get_cmap(P_total.dot(coords)), P_total


if __name__ == '__main__':
    import argparse
    import doctest
    import matplotlib.pyplot as plt
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    # parser.add_argument('--test', default=False, action='store_true', help='test the code')
    parser.add_argument('--pdb1', help='pdb filename of coordinates to permute', type=str)
    parser.add_argument('--pdb2', help='pdb filename of reference coordinates', type=str)
    parser.add_argument('--cmap', help='npy file of the target -- reference -- contact map. Multiple files can be given. In that case, the matrices will be concatenated as a block diagonal matrix', nargs='+', type=str)
    parser.add_argument('--niter', help='Number of iterations', type=int)
    parser.add_argument('--get_cmap', help='Compute the contact map from the given pdb and exit', type=str)
    parser.add_argument('--save_traj', help='Filename of an output dcd file to save optimization trajectory (optional)', type=str)
    args = parser.parse_args()

    # if args.test:
    #     doctest.testmod()
    #     sys.exit(0)

    if args.get_cmap is not None:
        coords = get_coords(args.get_cmap, 'pdbin')
        cmap = get_cmap(coords)
        basename = os.path.splitext(args.get_cmap)[0]
        numpy.save(f'{basename}_cmap.npy', cmap)
        sys.exit(0)

    coords1 = get_coords(args.pdb1, 'shuf')
    A = get_cmap(coords1)
    if args.pdb2 is not None:
        coords_ref = get_coords(args.pdb2, 'ref')
        B = get_cmap(coords_ref)
    else:
        cmaps = []
        for cmap in args.cmap:
            cmaps.append(numpy.load(cmap))
        B = block_diag(*cmaps)
    P = permoptim(A, B)
    p, n = P.shape
    print(f"X: {coords1.shape}")
    print(f"A: {A.shape}")
    print(f"B: {B.shape}")
    print(f"P: {P.shape}")
    if n < p:
        coords1 = add_coords(coords1, p - n)
    plt.matshow(A)
    plt.savefig('cmap_shuf.png')
    plt.clf()
    plt.matshow(B)
    plt.savefig('cmap_ref.png')
    plt.clf()
    if args.save_traj is not None:
        save_traj = True
    else:
        save_traj = False
    A_optim, P = permiter(coords1, B, n_step=args.niter,
                          save_traj=save_traj, topology=args.pdb1,
                          outtrajfilename=args.save_traj)
    coords_out = permute_coords(coords1, P, same=False)
    mask = zero_mask(coords_out)
    A_optim = get_cmap(coords_out)
    A_optim[mask, :] = 0.
    plt.matshow(A_optim)
    plt.savefig('cmap_optim.png')
    plt.clf()
    coords_out = coords_out[~mask]
    cmd.load_coords(coords_out, 'shuf')
    cmd.save('coords_optim.pdb', 'shuf')
