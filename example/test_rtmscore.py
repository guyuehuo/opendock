from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.scorer.vina import VinaSF
from opendock.scorer.rtmscore import rtmsf
from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
from opendock.scorer.constraints import rmsd_to_reference
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.minimizer import lbfgs_minimizer
from opendock.core import io
import sys

# define a flexible ligand object
ligand = LigandConformation(sys.argv[1])
# define the receptor object
receptor = ReceptorConformation(sys.argv[2],
                                ligand.init_heavy_atoms_coords)
#receptor.init_sidechain_cnfrs()

ref_lig_fpath = sys.argv[3]

# define scoring function
sf = VinaSF(receptor, ligand)
vs = sf.scoring()
print("Vina Score ", vs)

# rtmscore
id, rtmscore = rtmsf(lig=sys.argv[1], prot=sys.argv[2], reflig=sys.argv[3])
print("rtmscore:", rtmscore)

exit()
# ligand center
xyz_center = ligand._get_geo_center().detach().numpy()[0]
print("Ligand XYZ COM", xyz_center)

# define sampler
print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
mc = MonteCarloSampler(ligand, receptor, sf,
                       box_center=xyz_center,
                       box_size=[20, 20, 20],
                       random_start=True,
                       minimizer=lbfgs_minimizer,
                       )
init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
print("Initial Score", init_score)

# run mc sampling
mc._random_move()
mc.sampling(100)

# save ligand conformations
mc.save_traj("traj_saved_100.pdb")
