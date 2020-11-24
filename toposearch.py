#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-11-20 12:40:09 (UTC+0100)

import sys
import pymol.cmd as cmd
import scipy.spatial.distance as distance
import scipy.sparse.csgraph as csgraph
import numpy as np


class Topology(object):
    def __init__(self, coords, d_target=3.8, sigma=0.1):
        self.coords = coords
        self.d_target = d_target
        self.sigma = sigma
        self.adjmat = self.get_adjmat()
        self.sdmat, self.preds = csgraph.shortest_path(self.adjmat,
                                                       return_predecessors=True)

    def get_adjmat(self):
        pdist = distance.squareform(distance.pdist(self.coords))
        adjmat = (pdist - self.d_target)**2 / (2 * self.sigma**2)
        return adjmat

    def chain(self, start, stop, out=None):
        """
        Shortest path from start to stop
        """
        if out is None:
            out = []
        node = stop
        out.append(node)
        if node != start:
            node = self.preds[start, node]
            return self.chain(start, node, out)
        else:
            return out[::-1]

    def all_path_len(self):
        """
        Matrix for storing length of all possible path
        """
        n = len(self.coords)
        D = []
        for i in range(n - 1):
            sys.stdout.write(f"Computing shortest path {i+1}/{n-1}    \r")
            sys.stdout.flush()
            for j in range(i + 1, n):
                D.append(len(self.chain(i, j)))
        D = np.asarray(D)
        print()
        return D


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb')
    args = parser.parse_args()

    cmd.load(args.pdb, 'inpdb')
    cmd.remove('not name CA')
    coords = cmd.get_coords('inpdb')
    topology = Topology(coords)
    # D = topology.all_path_len()
