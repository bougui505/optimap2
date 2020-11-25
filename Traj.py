#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-11-25 21:05:02 (UTC+0100)

from pymol import cmd
import psico.fullinit
from psico.exporting import *


class Traj(object):
    def __init__(self, topology):
        """
        - pmlobject: pymol object
        """
        cmd.load(topology, 'traj')
        self.nframes = cmd.count_states()

    def append(self, coords):
        """
        Append a frame to the trajectory
        """
        cmd.create('traj', 'traj', self.nframes, -1)
        self.nframes += 1
        cmd.load_coords(coords, 'traj', state=self.nframes)

    def save(self, outfilename):
        cmd.save_traj(outfilename, 'traj')


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()
