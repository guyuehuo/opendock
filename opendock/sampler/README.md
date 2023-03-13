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
standard ```samplers``` are implemented, the ```MonteCarloSampler```, the ```GeneticAlgorithmSampler```
and the ```MinimizerSampler```. For both ```samplers```, the main method is ```sampling```, which 
defines number of steps. The sampling history is stored in the object (```ligand_scores_history_```, 
```ligand_cnfrs_history_``` and ```receptor_cnfrs_history_```). By using the ```cluster``` class, 
the lowest energy poses or receptor conformations are therefore selected. 

## How to implement user defined ```sampler```?

## How to combine ```samplers``` and ```scorers```?