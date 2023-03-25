#!/bin/bash

python=/d/Anaconda/envs/dgl/python

rec=example/4tmn/4tmn_protein_atom_noHETATM.pdbqt
lig=example/4tmn/4tmn_ligand.sdf
pose=example/4tmn/4tmn_448.pdbqt


$python test_clip.py $pose $rec $lig
#$python distance_constraints.py  $pose $rec
#$python outofbox_constraints.py $pose $rec
#python test_HETATM.py $pose $rec $hetatm
#$python test_HETATM.py $pose $rec
