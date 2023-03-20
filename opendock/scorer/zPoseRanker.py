

from opendock.scorer.scoring_function import BaseScoringFunction, ExternalScoringFunction
import os, sys
import subprocess as sp 
import pandas as pd
import torch
import shutil
import uuid


DTI_PY_BIN = "/user/zhengliangzhen/.conda/envs/protein/bin/python"
DTI_NPY_SCRIPT = "/user/zhengliangzhen/apps/dtiPipeline/run_decoys2npy.py"
DTI_SCR_SCRIPT = "/user/zhengliangzhen/apps/dtiPipeline/bin/inference.sh"


class zPoseRankerSF(ExternalScoringFunction):

    def __init__(self, receptor, ligand, **kwargs):
        super(zPoseRankerSF, self).__init__(receptor=receptor, ligand=ligand)

        self.version = kwargs.pop('version', 'v0.0.3')
        self.verbose = kwargs.pop('verbose', False)
    
    def _score(self, receptor_fpath=None, ligand_fpath=None):
        if self.tmp_dpath is None:
            self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
            os.makedirs(self.tmp_dpath, exist_ok=True)

        # npy file 
        cmd = f"{DTI_PY_BIN} {DTI_NPY_SCRIPT} --ligand {ligand_fpath} --receptor \
                {receptor_fpath} --output {self.tmp_dpath}/output"
        if self.verbose: print(f"Running cmd: {cmd}")
        job = sp.Popen(cmd, shell=True)
        job.communicate()

        # run score function
        cmd = f"{DTI_SCR_SCRIPT} {self.tmp_dpath}/output/output.npy {self.tmp_dpath}/output/scores.csv {self.version}"
        if self.verbose: print(f"Running cmd: {cmd}")
        job = sp.Popen(cmd, shell=True)
        job.communicate()

        # output file 
        self.output_csv_fpath = f"{self.tmp_dpath}/output/scores_{self.version}.csv"

        if os.path.exists(self.output_csv_fpath):
            scores = pd.read_csv(self.output_csv_fpath, header=0)['l_mean_prmsd'].values #.tolist()
        else:
            scores = [0, ] 

        return scores

    def scoring(self, ligand_cnfrs=None, receptor_cnfrs_list=None, remove_temp=True):

        if self.tmp_dpath is None:
            self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
            os.makedirs(self.tmp_dpath, exist_ok=True) 

        # generate receptor and ligand pdb file 
        if self.receptor_fpath is None:
            self.receptor_fpath = self._prepare_receptor_fpath(cnfrs_list=receptor_cnfrs_list)

        if self.ligand_fpath is None:
            self.ligand_fpath   = self._prepare_ligand_fpath(cnfrs=ligand_cnfrs)

        _scores = self._score(self.receptor_fpath, self.ligand_fpath)

        # remove temp dpath 
        if remove_temp:
            shutil.rmtree(self.tmp_dpath)

        return torch.Tensor(_scores).reshape((1, -1))


if __name__ == "__main__":

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)

    sf = zPoseRankerSF(receptor=receptor, ligand=ligand, verbose=True)
    print(sf._score(ligand_fpath=sys.argv[1], receptor_fpath=sys.argv[2]))
