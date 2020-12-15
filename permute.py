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
import IO


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


def shuffle_P(P, noise):
    p, n = P.shape
    A = numpy.identity(p)
    inds1 = numpy.random.choice(p, size=int(p * noise), replace=False)
    inds2 = numpy.random.choice(inds1, size=int(p * noise), replace=False)
    A[inds1, inds1] = 0.
    A[inds1, inds2] = 1.
    assert (A.sum(axis=0) == 1.).all()
    assert (A.sum(axis=1) == 1.).all()
    P = A.dot(P)
    return P


def permute_coords(coords, P, same=True, random=False, noise=1.):
    """
    Permute the coordinates using P
    If same is True, return a coordinates array with the same shape as the input coords
    - random: if True, randomize the permutations
    """
    p, n = P.shape
    if p < n and same:  # Works
        sel = (P.sum(axis=0) == 0)
        inds = numpy.where(sel)  # Coordinates not used
        P1 = numpy.zeros((n - p, n))
        P1[range(n - p), inds] = 1
        P = numpy.block([[P], [P1]])
    if random:
        P = shuffle_P(P, noise)
    assert (P.sum(axis=0) == 1.).all()
    assert (P.sum(axis=1) == 1.).all()
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
        # if n > p:
        #     B = block_diag(B, numpy.identity(n - p))
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

    def get_beta(self, rate50=0.1):
        """
        - rate50: ration of the total number of contact with a acceptance rate of .5
        """
        delta50 = self.B.sum() * 0.1
        beta = -numpy.log(0.5) / delta50
        return beta

    def iterate(self, n_epoch=10, n_iter=1000, alpha_target=0.234,
                save_traj=False, topology=None, outtrajfilename='permiter.dcd'):
        X_P, P = permute_coords(self.X, self.P, random=True)
        P_total = P.copy()
        A_optim = get_cmap(X_P)
        scores = []
        score_steps = []
        if save_traj:
            traj = Traj.Traj(topology)
        global_min = numpy.inf
        local_min = numpy.inf
        with open('permiter.log', 'w') as logfile:
            logfile.write(f"n_epoch: {n_epoch}\n")
            logfile.write(f"n_iter: {n_iter}\n\n")
            miniter = 0
            epoch = 0
            P0 = P.copy()
            X_P0 = X_P.copy()
            P_total0 = P_total.copy()
            A_optim0 = A_optim.copy()
            score0 = numpy.inf
            beta = 1.  # self.get_beta()
            alpha = 1.
            alpha_mean = 0.
            ds_mean = 0.
            while epoch < n_epoch:
                miniter += 1
                P = permoptim(A_optim[:self.n, :self.n], self.B, P[:self.p, :self.n])
                X_P, P = permute_coords(X_P[:self.n, :], P)
                P_total = P.dot(P_total)
                A_optim = get_cmap(X_P)
                if save_traj:
                    traj.append(X_P)
                mask = zero_mask(X_P)
                score = self.get_score(A_optim, mask)
                if score == 0.:
                    print()
                    print("Early stop")
                    break
                if score < local_min:
                    miniter = 0
                    if score < global_min:
                        global_min = score
                        P_total_best = P_total
                    local_min = score
                    P_lm = P.copy()
                    X_P_lm = X_P.copy()
                    P_total_lm = P_total.copy()
                    A_optim_lm = A_optim.copy()
                    score_lm = score.copy()
                    # X_P_restart = X_P
                    # P_restart = P
                scores.append(score)
                if miniter > n_iter:
                    P = P_lm
                    X_P = X_P_lm
                    P_total = P_total_lm
                    A_optim = A_optim_lm
                    score = score_lm
                    epoch += 1
                    score_steps.append(scores[-1])
                    local_min = numpy.inf
                    ds = score - score0
                    if not numpy.isinf(ds):
                        ds_mean = ds_mean + (ds - ds_mean) / epoch
                    else:
                        ds_mean = 0.
                    if ds_mean > 0.:
                        beta = -numpy.log(alpha_target) / ds_mean
                    alpha = min(numpy.exp(-beta * ds), 1.)
                    alpha_mean = alpha_mean + (alpha - alpha_mean) / epoch
                    if numpy.random.uniform() > alpha:  # reject
                        P = P0.copy()
                        X_P = X_P0.copy()
                        P_total = P_total0.copy()
                        A_optim = A_optim0.copy()
                        score = score0.copy()
                        logfile.write('\naccepted: 0')
                    else:
                        logfile.write('\naccepted: 1')
                    P0 = P.copy()
                    X_P0 = X_P.copy()
                    P_total0 = P_total.copy()
                    A_optim0 = A_optim.copy()
                    score0 = score.copy()
                    P += numpy.random.uniform(low=0., high=1., size=P.shape)
                    # P = shuffle_P(P, noise=1.)
                    logfile.write(f'\nalpha: {alpha:.3f}\nbeta: {beta:.3f}')
                logfile.write(f'\nepoch: {epoch}\nminiter: {miniter}\nscore: {score:.3f}')
                logfile.write('\n')
                sys.stdout.write(f'{epoch:4d}/{n_epoch:4d} {miniter:4d}/{n_iter:4d} {score:6.1f}/{global_min:6.1f} local_min: {local_min:6.1f} α: {alpha:1.1f} α_mean: {alpha_mean:1.1f} β: {beta:1.1g}\r')
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
    parser.add_argument('--n_epoch', help='Number of epochs', type=int)
    parser.add_argument('--n_iter', help='Minimizer iterations', type=int)
    parser.add_argument('--get_cmap', help='Compute the contact map from the given pdb and exit', type=str)
    parser.add_argument('--save_traj', help='Filename of an output dcd file to save optimization trajectory (optional)', type=str)
    parser.add_argument('--fasta', help='Fasta file with the sequence to write in the output pdb', nargs='+', type=str)
    parser.add_argument('--resids', help='column text file with the residue numbering', nargs='+', type=str)
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
    if args.save_traj is not None:
        save_traj = True
    else:
        save_traj = False
    permiter = Permiter(coords1, B)
    plt.matshow(permiter.A)
    plt.savefig('cmap_shuf.png')
    plt.clf()
    plt.matshow(permiter.B)
    plt.savefig('cmap_ref.png')
    plt.clf()
    A_optim, P = permiter.iterate(n_epoch=args.n_epoch,
                                  n_iter=args.n_iter,
                                  save_traj=save_traj, topology=args.pdb1,
                                  outtrajfilename=args.save_traj)
    plt.matshow(permiter.A_P)
    plt.savefig('cmap_optim.png')
    plt.clf()
    coords_out = permiter.X_P
    if args.fasta is not None:
        seqs = []
        for fasta in args.fasta:
            seq = IO.read_fasta(fasta)
            seqs.extend(seq)
        if len(seqs) < permiter.n:
            seqs.extend(['DUM', ] * (permiter.n - len(seqs)))
        seq = numpy.asarray(seqs)[~permiter.mask]
    else:
        seq = None
    if args.resids is not None:
        resids = []
        for resfile in args.resids:
            resids.extend(list(numpy.genfromtxt(resfile, dtype=int)))
        if len(resids) < permiter.n:
            resids.extend([0, ] * (permiter.n - len(resids)))
        resids = numpy.asarray(resids)[~permiter.mask]
    else:
        resids = None
    IO.write_pdb('shuf', coords_out, 'coords_optim.pdb', seq=seq, resids=resids)
