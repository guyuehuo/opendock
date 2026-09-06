import os, sys
from opendock.core.conformation import LigandConformation, ReceptorConformation
from opendock.sampler.minimizer import adam_minimizer, sgd_minimizer, lbfgs_minimizer
from opendock.sampler.minimizer import MinimizerSampler
from opendock.scorer.vina import VinaSF
from opendock.scorer.hybrid import HybridSF
from opendock.scorer.constraints import OutOfBoxConstraint
from opendock.core.io import write_ligand_traj
import numpy as np
from opendock.core.utils import merge_rec_hetatm

lig_file = sys.argv[1]
rec_file = sys.argv[2]

ligand = LigandConformation(lig_file)
ligand.parse_ligand()

print(len(sys.argv))
if len(sys.argv) == 4:
    hetatm_inp = sys.argv[3]
    rec_het_file = merge_rec_hetatm(rec_file, hetatm_inp)
    print("rec_het_file:", rec_het_file)
    receptor = ReceptorConformation(rec_het_file)
else:
    receptor = ReceptorConformation(sys.argv[2])
receptor.parse_receptor()
print("xyz:", receptor.init_rec_heavy_atoms_xyz.shape)

xyz_center = ligand._get_geo_center().detach().numpy()[0]
print("Ligand XYZ COM", xyz_center)

# constraints
cnstr = OutOfBoxConstraint(receptor, ligand, box_center=xyz_center,
                           box_size=[15, 15, 15], force=2.0)

print("OutOfBox scoring", cnstr.scoring())

# vina scoring function
sf1 = VinaSF(receptor, ligand)
vs = sf1.scoring()
print("Vina Score ", vs)

# combined scoring function
sf = HybridSF(receptor, ligand, scorers=[sf1, cnstr], weights=[0.5, 0.5])
vs = sf.scoring()
print("HybridSF Score ", vs)

# monte carlo
print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
mi = MinimizerSampler(ligand=ligand, receptor=receptor,
                      scoring_function=sf,
                      minimizer=adam_minimizer,
                      box_center=xyz_center,
                      box_size=[20, 20, 20]
                      )
init_score = mi._score(ligand.cnfrs_, receptor.cnfrs_)
print("Initial Score", init_score)
mi._random_move(ligand.cnfrs_, receptor.cnfrs_)
mi.sampling(nsteps=ligand.number_of_frames * 5000, init_lr=0.5,
            rounds=ligand.number_of_frames * 10)

#print("Final confrs: ", mi.ligand_cnfrs_his  tory_)
write_ligand_traj(mi.ligand_cnfrs_history_, ligand,
                  "traj_mini_full.pdb",
                  {"hybrid_score": mi.ligand_scores_history_})
