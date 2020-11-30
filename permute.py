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
from tsp_solver.greedy import solve_tsp


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


def topofix(coords, dist_ca=3.8, sigma_ca=0.1, mask=None):
    def get_topo(coords):
        pdist = distance.squareform(distance.pdist(coords))
        topo = numpy.exp(-(pdist - dist_ca)**2 / (2 * sigma_ca**2))
        return topo
    n = coords.shape[0]
    topo = get_topo(coords)
    if mask is not None:
        topo[mask] = 0.5
    col_ind = solve_tsp(1. - topo, endpoints=(0, n - 1))
    row_ind = numpy.arange(n)
    P = numpy.zeros((n, n))
    P[row_ind, col_ind] = 1.
    return P


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
    P[col_ind, row_ind] = 1.
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


def get_prob_swap(v):
    # expv = numpy.exp(v)
    # return expv / expv.sum()
    return v / v.sum()


class Permiter(object):
    def __init__(self, X, B):
        """
        X: coordinates
        B: target contact map
        """
        n = X.shape[0]
        p = B.shape[0]
        if n < p:
            X = add_coords(X, p - n)
        self.X = X
        self.B = B
        self.A = get_cmap(self.X)
        self.n = self.X.shape[0]
        self.p = self.B.shape[0]
        self.P = permoptim(self.A, self.B)
        print(f"X: {self.X.shape}")
        print(f"A: {self.A.shape}")
        print(f"B: {self.B.shape}")
        print(f"P: {self.P.shape}")

    def iterate(self, n_step=10000, restart=3, stop=10,
                save_traj=False, topology=None, outtrajfilename='permiter.dcd'):
        X_P, P = permute_coords(self.X, self.P)
        P_total = P.copy()
        A_optim = get_cmap(X_P)
        scores = []
        score_steps = []
        if save_traj:
            traj = Traj.Traj(topology)
        tsp_count = 0
        score_min = numpy.inf
        with open('permiter.log', 'w') as logfile:
            logfile.write(f"niter: {n_step}\n\n")
            for i in range(n_step):
                P = permoptim(A_optim[:self.n, :self.n], self.B, P[:self.p, :self.n])
                X_P, P = permute_coords(X_P[:self.n, :], P)
                P_total = P.dot(P_total)
                if save_traj:
                    traj.append(X_P)
                A_optim = get_cmap(X_P)
                mask = zero_mask(X_P)
                score = self.get_score(A_optim, mask)
                if score < score_min:
                    score_min = score
                    P_total_best = P_total
                    X_P_restart = X_P
                    P_restart = P
                scores.append(score)
                # if (scores[-1] == numpy.asarray(scores)).sum() > 3:
                stopcount = (numpy.isclose(scores[-1], scores[:-1], atol=1e-3, rtol=0)).sum()
                logfile.write(f'\nstep: {i}\nscore: {score:.3f}\nstopcount: {stopcount}')
                if stopcount >= restart:
                    score_steps.append(scores[-1])
                    # _, counts = numpy.unique(score_steps, return_counts=True)
                    # count = max(counts)
                    if stopcount > stop or score == 0.:
                        print("Early stop")
                        break
                    else:
                        logfile.write(f'\nTSP: 1')
                        # X_P = X_P_restart
                        # P = P_restart
                        # P_total = P_total_best
                        # A_optim = get_cmap(X_P)
                        P = topofix(X_P, mask=mask)
                        tsp_count += 1
                        # P = self.shuffle_P(P)
                logfile.write('\n')
                sys.stdout.write(f'{i+1}/{n_step} {score:10.3f}/{score_min:10.3f} n_tsp: {tsp_count}')
                sys.stdout.write('             \r')
                sys.stdout.flush()
        print()
        if save_traj:
            traj.save(outtrajfilename)
        # Store results in class
        self.P = P_total_best
        self.X_P = self.P.dot(self.X)
        self.mask = zero_mask(self.X_P)
        self.A_P = get_cmap(self.X_P)
        self.A_P[self.mask, :] = 0.
        self.A_P = self.A_P[:self.p, :self.p]
        self.X_P = self.X_P[~self.mask]
        return self.A_P, self.P

    def get_score(self, A, mask, per_col=False):
        score = ((A[:self.p][:, :self.p] - self.B)**2)[~mask[:self.p]].sum(axis=0)
        if per_col:
            return score
        else:
            return score.sum()

    def shuffle_P(self, P):
        """
        Shuffle rows of P
        """
        return P + numpy.random.uniform(size=P.shape)


if __name__ == '__main__':
    import argparse
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
    # p, n = P.shape
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
    permiter = Permiter(coords1, B)
    A_optim, P = permiter.iterate(n_step=args.niter,
                                  save_traj=save_traj, topology=args.pdb1,
                                  outtrajfilename=args.save_traj)
    plt.matshow(permiter.A_P)
    plt.savefig('cmap_optim.png')
    plt.clf()
    coords_out = permiter.X_P
    cmd.load_coords(coords_out, 'shuf')
    cmd.save('coords_optim.pdb', 'shuf')
