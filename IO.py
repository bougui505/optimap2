#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-12-01 16:23:48 (UTC+0100)

from pymol import cmd


def read_fasta(fasta_file):
    """
    read only 1 chain
    """
    aa1 = list("ACDEFGHIKLMNPQRSTVWY")
    aa3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    aa123 = dict(zip(aa1, aa3))
    # aa321 = dict(zip(aa3, aa1))
    with open(fasta_file) as fasta:
        seq = ''
        for line in fasta:
            if line[0] == '>':
                pass
            else:
                seq += line.strip()
    seq = [aa123[r] for r in seq]
    return seq


def write_pdb(obj, coords, outfilename, seq=None, resids=None, chains=None):
    cmd.load_coords(coords, obj)
    if seq is not None:
        myspace = {}
        myspace['seq_iter'] = iter(seq)
        cmd.alter(obj, 'resn=f"{seq_iter.__next__()}"', space=myspace)
    if resids is not None:
        myspace = {}
        myspace['resid_iter'] = iter(resids)
        cmd.alter(obj, 'resi=f"{resid_iter.__next__()}"', space=myspace)
    if chains is not None:
        myspace = {}
        myspace['chain_iter'] = iter(chains)
        cmd.alter(obj, 'chain=f"{chain_iter.__next__()}"', space=myspace)
    cmd.save(outfilename, selection=obj)
