#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-12-16 16:26:24 (UTC+0100)

import numpy as np


class metrics(object):
    def __init__(self, cmap, cmap_ref, t, t_ref):
        self.t = t
        self.t_ref = t_ref
        self._cmap = cmap
        self._cmap_ref = cmap_ref

    @property
    def cmap(self):
        return np.int_(self._cmap >= self.t)

    @property
    def cmap_ref(self):
        return np.int_(self._cmap_ref >= self.t_ref)

    @property
    def TP(self):
        return (self.cmap * self.cmap_ref).sum()

    @property
    def TN(self):
        cmap_n = 1 - self.cmap
        cmap_ref_n = 1 - self.cmap_ref
        return (cmap_n * cmap_ref_n).sum()

    @property
    def FP(self):
        cmap_ref_n = 1 - self.cmap_ref
        return (self.cmap * cmap_ref_n).sum()

    @property
    def FN(self):
        cmap_n = 1 - self.cmap
        return (cmap_n * self.cmap_ref).sum()

    @property
    def P(self):
        return self.TP + self.FN

    @property
    def N(self):
        return self.TN + self.FP

    @property
    def Se(self):
        return self.TP / self.P

    @property
    def Sp(self):
        return self.TN / self.N

    @property
    def MCC(self):
        return ((self.TP * self.TN) - (self.FP * self.FN)) / np.sqrt((self.TP + self.FP) * self.P * self.N * (self.TN + self.FN))


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()
