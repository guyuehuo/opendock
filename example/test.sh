#!/bin/bash

python=/home/wzc/anaconda3/envs/RTMScore/bin/python

rec_fpath=example/4tmn/4tmn_protein_atom_noHETATM.pdbqt
lig_fpath=example/4tmn/4tmn_448.pdbqt
ref_lig_fpath=example/4tmn/4tmn_ligand.sdf

#rec_fpath=example/1gpn/1gpn_protein_atom_noHETATM.pdbqt
#lig_fpath=example/1gpn/decoys/1gpn_404.pdbqt
#ref_lig_fpath=example/1gpn/1gpn_ligand.sdf

$python test_rtmscore.py $lig_fpath $rec_fpath $ref_lig_fpath

