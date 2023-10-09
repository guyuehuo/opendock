
from opendock.scorer.rtmscore import RtmscoreExtSF
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
import sys, os
import torch
import numpy as np


if __name__ == "__main__": 

    if len(sys.argv) < 2:
        print("python script.py ligand.pdbqt receptor.pdbqt")
        sys.exit(0)

    if os.path.exists(sys.argv[3]):
        sys.exit(0)

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    print(ligand.init_heavy_atoms_coords.shape)
    xyz_center = ligand.init_heavy_atoms_coords.mean(axis=1)
    receptor = ReceptorConformation(sys.argv[2], 
                                    torch.Tensor(xyz_center).reshape((1, 3)), 
                                    ligand.init_heavy_atoms_coords)

    sf = RtmscoreExtSF(receptor=receptor, ligand=ligand)
    sf.receptor_fpath = sys.argv[2]
    sf.ligand_fpath = sys.argv[1]
    sf.output_fpath = sys.argv[3]
    scores = sf.scoring().detach().numpy().ravel()

    #np.savetxt(sys.argv[3], scores, fmt="%.3f")

    