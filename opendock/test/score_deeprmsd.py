
from opendock.scorer.deeprmsd import DeepRmsdSF, CNN
from opendock.scorer.hybrid import HybridSF
from opendock.scorer.vina import VinaSF
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
import sys, os
import torch
import numpy as np
import uuid
import shutil
import subprocess as sp


if __name__ == "__main__": 

    if len(sys.argv) < 2:
        print("python script.py ligand.pdbqt receptor.pdbqt")
        sys.exit(0)

    if os.path.exists(sys.argv[3]):
        sys.exit(0)

    # make a temp dir
    temp_dpth = f"/tmp/deeprmsd_{uuid.uuid4().hex}"
    os.makedirs(temp_dpth, exist_ok=True)

    # split pdbqt files
    cmd = f"obabel {sys.argv[1]} -O {temp_dpth}/pose_.pdbqt -m"
    job = sp.Popen(cmd, shell=True)
    job.communicate()

    # n files
    nf = len(os.listdir(temp_dpth))

    scores = []

    try:
        for i in range(1, nf + 1):
            # define a flexible ligand object 
            lfpth = f"{temp_dpth}/pose_{i}.pdbqt"
            ligand = LigandConformation(lfpth)
            #print(ligand.init_heavy_atoms_coords.shape)
            xyz_center = ligand.init_heavy_atoms_coords.mean(axis=1)
            receptor = ReceptorConformation(sys.argv[2], 
                                            torch.Tensor(xyz_center).reshape((1, 3)), 
                                            ligand.init_heavy_atoms_coords)

            sf1 = VinaSF(receptor, ligand)
            vs1 = sf1.scoring().detach().numpy().ravel()[0]
            print("Vina Score ", vs1)

            # define scoring function
            sf2 = DeepRmsdSF(receptor, ligand)
            vs2 = sf2.scoring().detach().numpy().ravel()[0]
            print("DeepRMSD Score ", vs2)

            # combined scoring function
            sf = HybridSF(receptor, ligand, scorers=[sf1, sf2], weights=[0.8, 0.2])

            scores.append([vs1, vs2, vs1 * 0.5 + vs2 * 0.5])
            
            #print(scores)
        np.savetxt(sys.argv[3], np.array(scores), fmt="%.3f")

    except:
        print("===> bad docking pose")

    shutil.rmtree(temp_dpth)
    print("temp files are cleaned ...")

        