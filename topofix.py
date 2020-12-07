#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-12-07 09:55:44 (UTC+0100)

import sys
import scipy.optimize as optimize
import scipy.spatial.distance as distance
import numpy
import permute


def stdoutarr(A):
    numpy.savetxt(sys.stdout, A)


def get_topo(coords, dist_ca=3.8, sigma_ca=0.1):
    pdist = distance.squareform(distance.pdist(coords))
    topo = numpy.exp(-(pdist - dist_ca)**2 / (2 * sigma_ca**2))
    return topo


def topofix(A):
    n = A.shape[0]
    B = numpy.zeros_like(A)
    B[numpy.arange(n - 1), numpy.arange(1, n)] = 1.
    B[numpy.arange(1, n), numpy.arange(n - 1)] = 1.
    P = permute.permoptim(A, B)
    return P


def topofix_coords(coords):
    topo = get_topo(coords)
    P = topofix(topo)
    return P


if __name__ == '__main__':
    coords = numpy.loadtxt(sys.stdin)
    P = topofix_coords(coords)
    coords_P = P.dot(coords)
    topo_P = get_topo(coords_P)
    stdoutarr(topo_P)
