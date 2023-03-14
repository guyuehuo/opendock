
from opendock.scorer.scoring_function import BaseScoringFunction
from opendock.core.io import write_ligand_traj, write_receptor_traj
import uuid
import os, sys
import shutil
import subprocess as sp 
import torch


class OnionNetSFCTSF(BaseScoringFunction):

    def __init__(self, receptor=None, ligand=None, **kwargs):
        super().__init__(receptor=receptor, ligand=ligand)
        self.receptor = receptor
        self.ligand   = ligand
        self.python_exe = kwargs.pop("python_exe", "python")
        self.scorer_bin = os.path.abspath(kwargs.pop("scorer_bin", "scorer.py"))
        self.sfct_dpath = os.path.dirname(self.scorer_bin)

    def _prepare_files(self):
        # receptor pdbqt 
        self.receptor_fpath = os.path.join(self.temp_dpath, "receptor.pdb")
        if self.receptor.cnfrs_ is None:
            with open(self.receptor_fpath, 'w') as tf:
                for l in self.receptor.receptor_original_lines:
                    tf.write(l)
        else:
            write_receptor_traj(self.receptor.cnfrs_, self.receptor, 
                                self.receptor_fpath)
        
        # ligand pdbqt 
        self.ligand_fpath = os.path.join(self.temp_dpath, "ligand.pdb")
        if self.ligand.cnfrs_ is None:
            write_ligand_traj(self.ligand.init_cnfrs, self.ligand, 
                                self.ligand_fpath)
        else:
            write_ligand_traj(self.ligand.cnfrs_, self.ligand, 
                                self.ligand_fpath)
    
    def _run_sfct(self):

        """$PYTHON_ENV/bin/python $ZYDOCK_ROOT/tools/OnionNet-SFCT/scorer.py 
        -r $rec -l $out/$outfile -o $out/sfct_scores.txt 
        --model $ZYDOCK_ROOT/tools/OnionNet-SFCT/model/rf.model --ncpus 16"""
        outfile = os.path.join(self.temp_dpath, "sfct.txt")

        cmd = f"{self.python_exe} {self.scorer_bin} -r {self.receptor_fpath} \
                -l {self.ligand_fpath} -o {outfile} \
                --model {self.sfct_dpath}/model/rf.model --ncpus 1 --stype general"
        print(f"[INFO] running cmd {cmd}")
        job = sp.Popen(cmd, shell=True)
        job.communicate()

        if os.path.exists(outfile):
            with open(outfile) as lines:
                try:
                    score = [float(x.split()[-1]) for x in lines if "#" not in x][0]
                except IndexError:
                    score = 9.99
        else:
            print("[WARNING] failed to obtain sfct scores ...")
            score = 9.99
        
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


    def scoring(self, remove_temp=True) -> torch.Tensor:
        # make temp directory
        self.temp_dpath = f"/tmp/sfct_{str(uuid.uuid4().hex)}"
        os.makedirs(self.temp_dpath, exist_ok=True)

        # make files 
        self._prepare_files()

        # run scoring 
        score = self._run_sfct()

        # do clean-up
        if remove_temp:
            shutil.rmtree(self.temp_dpath)

        return torch.Tensor([[score, ]])


if __name__ == "__main__":

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)
    #receptor.init_sidechain_cnfrs()

    sf = OnionNetSFCTSF(receptor, ligand, 
                        python_exe="/share/zhengliangzhen/apps/zydock/python_env/docking/bin/python3.6", 
                        scorer_bin="/share/zhengliangzhen/apps/zydock/tools/OnionNet-SFCT/scorer.py")
    score = sf.scoring(remove_temp=False)
    print("SFCT score ", score)
