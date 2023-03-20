
from opendock.scorer.scoring_function import ExternalScoringFunction
from opendock.core.io import write_ligand_traj, write_receptor_traj
import uuid
import os, sys
import shutil
import subprocess as sp 
import torch


SFCT_PY_BIN = "/share/zhengliangzhen/apps/zydock/python_env/docking/bin/python3.6"
SFCT_PY_SCRIPT = "/share/zhengliangzhen/apps/zydock/tools/OnionNet-SFCT/scorer.py"


class OnionNetSFCTSF(ExternalScoringFunction):
    def __init__(self, receptor=None, ligand=None, **kwargs):
        super(OnionNetSFCTSF, self).__init__(receptor=receptor, ligand=ligand)

        self.python_exe = kwargs.pop("python_exe", SFCT_PY_BIN)
        self.scorer_bin = kwargs.pop("scorer_bin", SFCT_PY_SCRIPT)
        self.sfct_dpath = os.path.dirname(self.scorer_bin)
        self.verbose = kwargs.pop("verbose", False)
    
    def _run_sfct(self):
        outfile = os.path.join(self.tmp_dpath, "sfct.txt")

        cmd = f"{self.python_exe} {self.scorer_bin} -r {self.receptor_fpath} \
                -l {self.ligand_fpath} -o {outfile} \
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
    
    def score_cnfrs(self, ligand_cnfrs=None, receptor_cnfrs = None):
        scores = []

        if ligand_cnfrs is not None and receptor_cnfrs is not None:
            assert len(ligand_cnfrs) == len(receptor_cnfrs)

            for _lcnfr, _rcnfr in zip(ligand_cnfrs, receptor_cnfrs):
                self.ligand.cnfrs_ = [_lcnfr]
                self.receptor.cnfrs_ = _rcnfr

                _score = self.scoring().detach().numpy().ravel()[0]
                scores.append(_score)
        elif ligand_cnfrs is not None and receptor_cnfrs is None:
            for _lcnfr in ligand_cnfrs:
                self.ligand.cnfrs_ = [_lcnfr]
                _score = self.scoring().detach().numpy().ravel()[0]
                scores.append(_score)
        else:
            for _rcnfr in receptor_cnfrs:
                self.receptor.cnfrs_ = _rcnfr
                _score = self.scoring().detach().numpy().ravel()[0]
                scores.append(_score)
        
        return torch.Tensor(scores).reshape((-1, 1))

    def scoring(self, ligand_cnfrs=None, 
                receptor_cnfrs_list=None, 
                remove_temp=True) -> torch.Tensor:
        # make temp directory
        self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
        os.makedirs(self.tmp_dpath, exist_ok=True)

        # generate receptor and ligand pdb file 
        if self.receptor_fpath is None:
            self.receptor_fpath = self._prepare_receptor_fpath(cnfrs_list=receptor_cnfrs_list)
        if self.ligand_fpath is None:
            self.ligand_fpath   = self._prepare_ligand_fpath(cnfrs=ligand_cnfrs)

        # run scoring 
        score = self._run_sfct()

        # do clean-up
        if remove_temp:
            shutil.rmtree(self.tmp_dpath)

        return torch.Tensor(score).reshape((1, -1))


if __name__ == "__main__":

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)

    sf = OnionNetSFCTSF(receptor, ligand, 
                        python_exe=SFCT_PY_BIN, 
                        scorer_bin=SFCT_PY_SCRIPT, 
                        verbose=True)
    score = sf.scoring(remove_temp=True)
    print("SFCT score ", score)
