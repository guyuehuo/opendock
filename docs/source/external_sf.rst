.. _external_scoring_function:


Step-by-step external scoring function implementation
=====================================================

External scoring functions could be traditional or machine learning-based or deep learning-based scoring methods for protein-ligand binding prediction.
These scoring functions could be used as post-scoring methods for docking pose re-ranking after sampling (and clustering).

Some of the common scoring functions could be:

.. code-block:: bash

  OnionNet-SFCT
  RTMscore
  X-score
  zPoseScore

Or any other scoring functions that use the receptor and the docking pose as input. 
In the following part, we demostrate how external scoring functions are defined. 

1. OnionNet-SFCT
----------------------
OnionNet-SFCT is a machine learning-based scoring function that predicts the ligand docking pose RMSD (with regard to its native docking pose).
As a docking pose scoring function correction term, when it is combined with Vinascore (``VinaSF``), it is able to improve the docking success rate and virtual screening enrichment. 
The detail explanation of the scoring function could be found in this paper (https://doi.org/10.1093/bib/bbac051).  

Here is an example of the scoring function implemention.

Firstly, several modules should be imported for creating files, save intermediate results and call subprocess. 

.. code-block:: bash

    from opendock.scorer.scoring_function import ExternalScoringFunction
    from opendock.core.io import write_ligand_traj, write_receptor_traj
    import uuid
    import os, sys
    import shutil
    import subprocess as sp 
    import torch

Next, make sure you have already installed OnionNet-SFCT (https://github.com/zhenglz/OnionNet-SFCT.git) and could get correct results by running the examples. Supposing you have installed the package in your directory and the scoring script is defined by environment variable SFCT_PY_SCRIPT, then you may further define your python environment (SFCT_PY_SCRIPT).

.. code-block:: bash

    SFCT_PY_BIN = "/share/zhengliangzhen/apps/zydock/python_env/docking/bin/python3.6"
    SFCT_PY_SCRIPT = "/share/zhengliangzhen/apps/zydock/tools/OnionNet-SFCT/scorer.py"

.. note:: 

    You may change the content of SFCT_PY_BIN and SFCT_PY_SCRIPT according to your installation of OnionNet-SFCT. 

Now, define a scoring function class inherit from the ``ExternalScoringFunction`` class (defined in ``opendock/scorer/scoring_function.py``). This base scorer takes the receptor and the ligand objects as input, it could invoke ``subprocess`` to execute commond line and read the output file of the external tool.

In the following code block, the method ``_score`` executes the command line and parse the output file. 

.. code-block:: bash

    class OnionNetSFCTSF(ExternalScoringFunction):
        def __init__(self, receptor=None, ligand=None, **kwargs):
            super(OnionNetSFCTSF, self).__init__(receptor=receptor, ligand=ligand)
    
            self.python_exe = kwargs.pop("python_exe", SFCT_PY_BIN)
            self.scorer_bin = kwargs.pop("scorer_bin", SFCT_PY_SCRIPT)
            self.sfct_dpath = os.path.dirname(self.scorer_bin)
            self.verbose = kwargs.pop("verbose", False)
    
        def _score(self, receptor_fpath = None, ligand_fpath = None, **kwargs):
            outfile = os.path.join(self.tmp_dpath, "sfct.txt")
    
            cmd = f"{self.python_exe} {self.scorer_bin} -r {receptor_fpath} \
                    -l {ligand_fpath} -o {outfile} \
                    --model {self.sfct_dpath}/model/rf.model --ncpus 1 --stype general"
            if self.verbose: 
                print(f"[INFO] running sfct scoring cmd {cmd}")
            job = sp.Popen(cmd, shell=True)
            job.communicate()
    
            if os.path.exists(outfile):
                with open(outfile) as lines:
                    try:
                        score = [float(x.split()[-1]) for x in lines if "#" not in x]
                    except IndexError:
                        score = [9.99]
            else:
                print("[WARNING] failed to obtain sfct scores ...")
                score = [9.99]
            
            return score
    
        def score_cnfrs(self, ligand_cnfrs=None, receptor_cnfrs_list = None):
            # make temp directory
            if self.tmp_dpath is None:
                self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
                os.makedirs(self.tmp_dpath, exist_ok=True)
    
            scores = []
    
            if ligand_cnfrs is not None and receptor_cnfrs_list is not None:
                assert len(ligand_cnfrs) == len(receptor_cnfrs_list)
    
                for _lcnfr, _rcnfr in zip(ligand_cnfrs, receptor_cnfrs_list):
                    self.ligand.cnfrs_ = [_lcnfr]
                    self.receptor.cnfrs_ = _rcnfr
                    _score = self.scoring().detach().numpy().ravel()[0]
                    scores.append(_score)
            elif ligand_cnfrs is not None and receptor_cnfrs_list is None:
                for _lcnfr in ligand_cnfrs:
                    self.ligand.cnfrs_ = [_lcnfr]
                    _score = self.scoring().detach().numpy().ravel()[0]
                    scores.append(_score)
            else:
                for _rcnfr in receptor_cnfrs_list:
                    self.receptor.cnfrs_ = _rcnfr
                    _score = self.scoring().detach().numpy().ravel()[0]
                    scores.append(_score)
            
            return torch.Tensor(scores).reshape((-1, 1))


In another example, a hybrid scoring method based on OnionNet-SFCT+Vinascore is implemented.

.. note::

    The parameter alpha is a linear weight factor for the combination of Vinascore with OnionNet-SFCT. It is suggested to use alpha = 0.5 in most of the docking tasks and use alpha = 0.8 in virtual screening tasks.

.. code-block:: bash 

    class SFCTVinaSF(OnionNetSFCTSF):
        def __init__(self,
                     receptor = None,
                     ligand = None,
                     weight_alpha: float = 0.8,
                     ):
            # inheritant from base class
            super(SFCTVinaSF, self).__init__(receptor, ligand)
            self.weight_alpha = weight_alpha # the vina score weight
        
        def scoring(self):
            _vina_sf = VinaSF(ligand=self.ligand, 
                              receptor=self.receptor)
    
            return _vina_sf.scoring() * self.weight_alpha + \
                   self.scoring() * (1 - self.weight_alpha)

Here in the following code block, try to run the external scoring function with sampling.

.. code-block:: bash 

    if __name__ == "__main__":
        if len(sys.argv) < 2:
            print("usage: onnetsfct.py protein.pdbqt ligand.pdbqt output tag")
            sys.exit(0)
    
        if os.path.exists(sys.argv[3]):
            print(f"find previous output {sys.argv[3]}, exit now!!!")
            sys.exit(0)
    
        from opendock.core.conformation import ReceptorConformation
        from opendock.core.conformation import LigandConformation
    
        # define a flexible ligand object 
        ligand = LigandConformation(sys.argv[1])
        xyz_center = ligand._get_geo_center().detach().numpy()[0]
        print("Ligand XYZ COM", xyz_center) 
        receptor = ReceptorConformation(sys.argv[2], 
                                        xyz_center,
                                        ligand.init_heavy_atoms_coords)
    
        sf = OnionNetSFCTSF(receptor, ligand, 
                            python_exe=SFCT_PY_BIN, 
                            scorer_bin=SFCT_PY_SCRIPT, 
                            verbose=True)
        score = sf.scoring(remove_temp=True)
        print("SFCT score ", score)
    
        tf = open(sys.argv[3], 'w')
        try:
            tag = sys.argv[4]
        except:
            tag = "decoy"
    
        score = score.detach().numpy().ravel()[0] 
        tf.write(f'{tag},{score:.3f}\n')
        tf.close()
                                                       
