# Scoring functions in OpenDock framework. 
The scoring functions are the core components of the molecular modelling. 
In general, the scoring functions could be classified into different groups, 
such as knowledge-based methods (such as drugscore, chemscore), and emperical 
scoring functions (such as AutoDock Vina score), as well as machine-learning 
or deep learning scoring functions. 

Currently, the following scoring functions are implemented: Vinascore, DeepRMSD, 
ContactMap and DistanceMap based scoring functions, as well as distance-constrained
scoring functions. The above functions both could be computed from the receptor
and ligand heavy atoms' coordinates, thus are differentiable and could be optimized 
by minimizers (such as ```Adam```, ```SGD``` and ```LBFGS```) defined in the ```sampler``` 
module.

Here in this framework, it is supported to use external scoring functions, or user 
defined scoring functions. For example, the external scoring function ```RTMscore``` 
is implemented, but its input is ```receptor``` pdb file and ```ligand``` pdb file. 
Therefore, the scoring function could not work with ```minimizers```. However, 
it could be used for ```MonteCarloSampler``` and ```GeneticAlogrithmSampler``` but 
without ```minimizers```. 

## How to add an user-defined scoring function?
Bascially, the user-defined scoring function can be written as a child class of the 
```BaseScoringFunction```. This class should input the ```LigandConformation``` and 
the ```ReceptorConformation``` classes, and other ```kwargs```. It should have a method
called ```scoring()```, which overwrites the original method in the base class. 

In this ```scoring()``` method, firstly, the heavy atom coordinates of both the ```ligand```
and ```receptor``` objects should be updated using ```cnfrs2xyz()``` provided by either the
```LigandConformation``` or ```ReceptorConformation``` classes. For example, you use the 
number of heavy-atom contacts as the scoring function, you may therefore define the scoring
function as follows:

    from opendock.scorer.scoring_function import BaseScoringFunction

    # define user defined scoring function
    class ContactNumberSF(BaseScoringFunction):
        def __init__(self, receptor, ligand, **kwargs):
            super(ContactNumberSF, self).__init__(receptor=receptor, ligand=ligand)

            self.contact_cutoff = kwargs.pop('contact_cutoff', 5.0)

After which, then you write a ```scoring``` method: 

        def scoring(self):
            # update heavy atom coordinates
            self.ligand.cnfrs2xyz(self.ligand.cnfrs_)
            self.receptor.cnfrs2xyz(self.receptor.cnfrs_)

            # calculate distance matrix 
            distance_matrix = self.generate_distance_matrix()

            # contacts 
            contacts = torch.sum(distance_matrix <= self.contact_cutoff)

            return contacts

## How to implement the distance constraints in molecular docking?
Sometimes, you want to restrict the distance between specific atoms 
because the distance information is a prior condition. In this case, 
firstly we need to identify the atoms' heavy atom indices with the
help of the ```AtomSelection``` class. For example, the following code demonstrates
how to find the indices of specific atoms in both ```ligand``` and ```receptor``
objects. 

    from opendock.core.asl import AtomSelection

    asl = AtomSelection(molecule=receptor)
    indices_r = asl.select_atom(atomnames=['C,O,N,CA',], chains=['A'], residx=['120-122'])
    print(indices_r, receptor.dataframe_ha_.head())

    asl = AtomSelection(molecule=ligand)
    indices_l = asl.select_atom(atomnames=['N2,C13',])
    print(indices_l, ligand.dataframe_ha_.head())

After the selection, distance constraints are defined as ```scorer```: 

    # constraints
    cnstr = DistanceConstraintSF(receptor, ligand, 
                                 grpA_ha_indices=indices_r, 
                                 grpB_ha_indices=indices_l, 
                                 )
    print(cnstr.scoring())

This way, the distance constraint could be viewed as a scoring function, and combined 
with other scoring functions. This distance constraint is differentiable thus could 
work with ```sampler``` equiped with ```minimizer```. 

## Implement external scoring functions in ```opendock```
The following external scoring functions are implemented:

    OnionNet-SFCT (By Zheng, et al. in this paper: Improving protein–ligand docking and screening accuracies by incorporating a scoring function correction term)
    RTMscore (By Shen, et al in this paper: Boosting Protein–Ligand Binding Pose Prediction and Virtual Screening Based on Residue–Atom Distance Likelihood Potential and Graph Transformer)
    zPoseRanker (Zelixir's ligand pose rmsd prediction model used in CASP15)

For user defined external scoring functions, some key functions (```_score``` and ```score_cnfrs```) 
are defined by the user. The ```_score``` takes the protein (receptor) and the compound (ligand) 
files as input, and runs the related methods or commands (using ```subprocess.Popen```) and generates
related output files, which are further parsed by the user defined methods or codes. This function 
returns a list of floats (the scoring values, generally the lower the better.)

### OnionNet-SFCT
This scoring function is a machine-learning model trained with large-scale protein-ligand docking
decoys to predict the decoy deviation to the native pose. The scoring function is a correction term 
to the traditional scoring function (such as AutoDock Vina or other scoring functions). Here, we 
implement the scoring function by calling it with ```subprocess.Popen``` in python, and then read 
the output file generated by OnionNet-SFCT, and added to Vina scoring with a weight parameter. The 
weight of OnionNet-SFCT is suggested to be 0.8. To use this scoring function, it is required to 
install the scoring function as suggested in its github link: https://www.github.com/zhenglz/OnionNet-SFCT.git
Here is an example of calling this scoring function:

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation

    SFCT_PY_BIN = "/share/zhengliangzhen/apps/zydock/python_env/docking/bin/python3.6"
    SFCT_PY_SCRIPT = "/share/zhengliangzhen/apps/zydock/tools/OnionNet-SFCT/scorer.py"

    # define a flexible ligand object 
    ligand = LigandConformation("decoy_01.pdbqt")
    receptor = ReceptorConformation("receptor.pdbqt"), 
                                    ligand.init_heavy_atoms_coords)

    sf = OnionNetSFCTSF(receptor, ligand, 
                        python_exe=SFCT_PY_BIN, 
                        scorer_bin=SFCT_PY_SCRIPT)
    score = sf.scoring(remove_temp=True)
    print("SFCT score ", score)

The parameter ```SFCT_PY_BIN``` is the python binary executable file installed following the
instructions in the OnionNet-SFCT github README file.

### RTMscore
This scoring function takes the ligand poses and the protein structure as the input files, 
and return the scoring values (large the value, better the pose). Here in our implementation,
we multiply -1 to the scoring value to make lower value meaning better pose. To use this scoring
function.