#!/bin/bash

rec=example/1tae/1tae_protein_noHETATM.pdbqt
lig=example/1tae/1tae_NAD.pdbqt
pose=example/1tae/1tae_NAD.pdbqt
hetatm=example/1tae/HETATMs_pdbqt


#$python distance_constraints.py  $pose $rec
#$python outofbox_constraints.py $pose $rec
python test_HETATM.py $pose $rec $hetatm
#$python test_HETATM.py $pose $rec
