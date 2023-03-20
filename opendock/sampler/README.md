# ```Sampler``` defines the sampling function for the ligand pose and receptor sidechains. 

## Basic logic of the ```sampler``` 
The sampling objects takes the ligand and the receptor (sidechains) conformations as the 
"variables" to modify (mutating, changing, modifying or whatever other ways of manipulation). 
The advantage of the sampling the conformations (cnfr, 6+k torch tensors for the ligand, and 
list of torch tensors for the receptor side chains in the binding pocket) is that only a 
small number of degrees of freedom to deal with, and the molecule's structure integrity 
could be remained. That's to say, the molecule could looks reasonable during the sampling. 
Therefore, it is recommended to operate the ligand pose and the receptor sidechain orientations
with the conformation vectors (```cnfrs```). 
A ```sampler``` requires at least three components: the "scoring function" (```scorer```), 
the ligand and the receptor objects. The ```sampler``` keeps modifying the ```cnfrs```, and 
then evaluates the scores or energies by the ```scorer```. In this framework, at least three 
standard ```samplers``` are implemented:
    
    MonteCarloSampler
    GeneticAlgorithmSampler
    BayersianOptimizerSampler
    ParticleSwarmOptimizer
    MinimizerSampler

For both ```samplers```, the main method is ```sampling```, which 
defines number of steps. The sampling history is stored in the object (```ligand_scores_history_```, 
```ligand_cnfrs_history_``` and ```receptor_cnfrs_history_```). By using the ```cluster``` class, 
the lowest energy poses or receptor conformations are therefore selected. 

## How to implement user defined ```sampler```?
A ```sampler``` should have at least one core function ```sampling``` (which should be kept as it is).
Multiple steps are defined in the ```sampling``` function. For ```sampler```, the ```receptor``` 
object, the ```ligand``` object and the ```scoring_function``` object are key parameters or arguments. 

## How to combine ```samplers``` and ```scorers```?
In the following example, the usage of the sampler is explained in details. 

    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand=ligand, 
                           receptor=receptor, 
                           scoring_function=sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=adam_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score) 
    
The ```sampler``` here is an object of the ```MonteCarloSampler```, which requires the 
receptor, ligand, and scoring function objects. 