import os, sys
from opendock.core.receptor import Receptor
from opendock.core.conformation import LigandConformation
from opendock.sampler.minimizer import lbfgs_minimizer, adam_minimizer, sgd_minimizer
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.core.asl import AtomSelection
from opendock.scorer.vina import VinaSF 
from opendock.scorer.hybrid import HybridSF 
from opendock.scorer.constraints import DistanceConstraintSF


ligand = LigandConformation(sys.argv[1])
ligand.parse_ligand()

receptor = Receptor(sys.argv[2])
receptor.parse_receptor()

asl = AtomSelection(molecule=receptor)
indices_r = asl.select_atom(atomnames=['OH',], chains=['A'], residx=['130'])
print(indices_r, receptor.dataframe_ha_.head())

asl = AtomSelection(molecule=ligand)
indices_l = asl.select_atom(atomnames=['O1',])
print(indices_l, ligand.dataframe_ha_.head())

xyz_center = ligand._get_geo_center().detach().numpy()[0]
print("Ligand XYZ COM", xyz_center)

# constraints
cnstr = DistanceConstraintSF(receptor, ligand, 
                                grpA_ha_indices=indices_r, 
                                grpB_ha_indices=indices_l, 
                                constraint='upper', 
                                bounds=[3.0, ]
                                )
print(cnstr.scoring())

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
mc = MonteCarloSampler(ligand, receptor, sf, 
                        box_center=xyz_center, 
                        box_size=[20, 20, 20], 
                        random_start=True,
                        minimizer=lbfgs_minimizer,
                        )
init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
print("Initial Score", init_score)
mc._random_move()

# initialize GA
GA = GeneticAlgorithmSampler(ligand, receptor, sf, box_center=xyz_center, 
                                box_size=[20, 20, 20], minimizer=sgd_minimizer, n_pop=10)
#GA._initialize()
GA.sampling(n_gen=10)
