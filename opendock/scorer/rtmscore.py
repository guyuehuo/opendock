import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import uuid
import shutil
import MDAnalysis as mda
from torch.utils.data import DataLoader
try:
    from opendock.scorer.RTMScore.RTMScore.data.data \
        import VSDataset
    from opendock.scorer.RTMScore.RTMScore.model.utils \
        import collate, run_an_eval_epoch
    from opendock.scorer.RTMScore.RTMScore.model.model2 \
        import RTMScore, DGLGraphTransformer 
except:
    RTMscore, DGLGraphTransformer = None, None
    VSDataset, collate, run_an_eval_epoch = None, None, None

import torch.multiprocessing
from opendock.scorer.RTMScore.utils import obabel
from opendock.scorer.scoring_function import BaseScoringFunction, ExternalScoringFunction


torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.abspath(__file__).replace("rtmscore.py",".."))

_current_dpath = os.path.dirname(os.path.abspath(__file__))
RTMScore_Model = os.path.join(_current_dpath, "RTMScore/trained_models/rtmscore_model1.pth")

args = {}
args["batch_size"] = 128
args["dist_threhold"] = 5
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args["num_workers"] = 1
args["num_node_featsp"] = 41
args["num_node_featsl"] = 41
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.10


# External RTMscore package
PACKAGE_DPATH = "/share/zhengliangzhen/apps/RTMScore-main"
OBABEL     = os.path.join(PACKAGE_DPATH, "envs/rtmscore/bin/obabel")
RTM_PY_EXE = os.path.join(PACKAGE_DPATH, "envs/rtmscore/bin/python")


def rtmsf(prot, lig, modpath=RTMScore_Model,
            cut=10.0,
            gen_pocket=True,
            reflig=None,
            atom_contribution=False,
            res_contribution=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
            params_dict=args
            ):
    """
    prot: The input protein file ('.pdb')
    lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
    modpath: The path to store the pre-trained model
    gen_pocket: whether to generate the pocket from the protein file.
    reflig: The reference ligand to determine the pocket.
    cut: The distance within the reference ligand to determine the pocket.
    atom_contribution: whether the decompose the score at atom level.
    res_contribution: whether the decompose the score at residue level.
    explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
    use_chirality: whether to adopt the information of chirality to represent the molecules.
    parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
    kwargs: other arguments related with model
    """
    tmp_directory = f"/tmp/rtmscore_{str(uuid.uuid4().hex)}"

    print('prot:',prot)
    if not prot.endswith(".pdb"):
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory, exist_ok=True)
    
        out = os.path.join(tmp_directory, "receptor.pdb")
        obabel(prot, out)
        prot = out
    print('lig:',lig)
    if not lig.endswith(".mol2"):
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory, exist_ok=True)

        out = os.path.join(tmp_directory, "ligand.mol2")
        obabel(lig, out)
        lig = out

    if not reflig.endswith(".mol2"):
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory, exist_ok=True) 

        out = out = os.path.join(tmp_directory, "refer_ligand.mol2")
        obabel(reflig, out)
        reflig = out
    print('lig:',lig)
    data = VSDataset(ligs=lig,
                     prot=prot,
                     cutoff=cut,
                     gen_pocket=gen_pocket,
                     reflig=reflig,
                     explicit_H=explicit_H,
                     use_chirality=use_chirality,
                     parallel=parallel)

    test_loader = DataLoader(dataset=data,
                             batch_size=params_dict["batch_size"],
                             shuffle=False,
                             num_workers=params_dict["num_workers"],
                             collate_fn=collate)

    ligmodel = DGLGraphTransformer(in_channels=params_dict["num_node_featsl"],
                                   edge_features=params_dict["num_edge_featsl"],
                                   num_hidden_channels=params_dict["hidden_dim0"],
                                   activ_fn=th.nn.SiLU(),
                                   transformer_residual=True,
                                   num_attention_heads=4,
                                   norm_to_apply='batch',
                                   dropout_rate=0.15,
                                   num_layers=6
                                   )

    protmodel = DGLGraphTransformer(in_channels=params_dict["num_node_featsp"],
                                    edge_features=params_dict["num_edge_featsp"],
                                    num_hidden_channels=params_dict["hidden_dim0"],
                                    activ_fn=th.nn.SiLU(),
                                    transformer_residual=True,
                                    num_attention_heads=4,
                                    norm_to_apply='batch',
                                    dropout_rate=0.15,
                                    num_layers=6
                                    )

    model = RTMScore(ligmodel, protmodel,
                     in_channels=params_dict["hidden_dim0"],
                     hidden_dim=params_dict["hidden_dim"],
                     n_gaussians=params_dict["n_gaussians"],
                     dropout_rate=params_dict["dropout_rate"],
                     dist_threhold=params_dict["dist_threhold"]).to(params_dict['device'])

    checkpoint = th.load(modpath, map_location=th.device(params_dict['device']))
    model.load_state_dict(checkpoint['model_state_dict'])

    preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=params_dict['dist_threhold'],
                              device=params_dict['device'])

    # remove temporary directory
    shutil.rmtree(tmp_directory)

    return list(np.array(preds).ravel()) * -1.0 #th.Tensor(np.array(preds).reshape((-1, 1)))


class RtmscoreSF(ExternalScoringFunction):
    def __init__(self, receptor = None, ligand = None, **kwargs):
        super(RtmscoreSF, self).__init__(receptor=receptor, ligand=ligand)

        self.receptor = receptor
        self.ligand = ligand

        self.tmp_dpath = None
        self.receptor_fpath = None
        self.ligand_fpath   = None
    
    def _score(self, receptor_fpath=None, ligand_fpath=None):
        _scores = rtmsf(prot=receptor_fpath,
                        lig=ligand_fpath,
                        modpath=RTMScore_Model,
                        cut=10.0,
                        gen_pocket=True,
                        reflig=ligand_fpath,
                        explicit_H=False,
                        use_chirality=True,
                        parallel=False,
                        params_dict=args
                        )
    
        return _scores
    
    def score_cnfrs(self, ligand_cnfrs, receptor_cnfrs_list):

        scores = []
        if len(ligand_cnfrs) == len(receptor_cnfrs_list):
            for _lcnfrs, _rcnfrs in zip(ligand_cnfrs, receptor_cnfrs_list):
                self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
                os.makedirs(self.tmp_dpath, exist_ok=True) 

                _rec_fpath = self._prepare_receptor_fpath([_rcnfrs, ])[0]
                _lig_fpath = self._prepare_ligand_fpath([_lcnfrs, ])

                _scores = self._score(_rec_fpath, _lig_fpath)
                scores.append(_scores)

                # clean temporary files
                shutil.rmtree(self.tmp_dpath)
        else:
            for _lcnfrs in ligand_cnfrs:
                self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
                os.makedirs(self.tmp_dpath, exist_ok=True) 

                _rec_fpath = self._prepare_receptor_fpath([receptor_cnfrs_list[0], ])[0]
                _lig_fpath = self._prepare_ligand_fpath([_lcnfrs, ])

                _scores = self._score(_rec_fpath, _lig_fpath)
                scores.append(_scores)

                # clean temporary files
                shutil.rmtree(self.tmp_dpath)
        
        return th.Tensor(scores)
        

class RtmscoreExtSF(RtmscoreSF):

    def __init__(self, receptor = None, ligand = None, **kwargs):
        super(RtmscoreExtSF, self).__init__(receptor=receptor, ligand=ligand)

        self.receptor = receptor
        self.ligand = ligand

        self.tmp_dpath = None
        self.receptor_fpath = None
        self.ligand_fpath   = None
    
    def _score(self, receptor_fpath=None, ligand_fpath=None):
        rtm_out_fpath = f"{self.tmp_dpath}/rtmscore.csv" 

        if not os.path.exists(rtm_out_fpath):
            # convert protein
            obabel(receptor_fpath, f'{self.tmp_dpath}/receptor.pdb')

            # convert ligand
            obabel(ligand_fpath, f'{self.tmp_dpath}/docked_ligands.sdf')

            # convert ligand files
            cmd = f'{OBABEL} {ligand_fpath} -O {self.tmp_dpath}/pocket_.sdf -m'
            self._run_cmd(cmd)

            rmt_script = os.path.join(PACKAGE_DPATH, "rtmscore.py")
            cmd = [RTM_PY_EXE, rmt_script, f"-p {self.tmp_dpath}/receptor.pdb",
                f"-l {self.tmp_dpath}/docked_ligands.sdf", "-gen_pocket",
                f"-rl {self.tmp_dpath}/pocket_1.sdf", f"-o {self.tmp_dpath}/rtmscore",
                f"-m {PACKAGE_DPATH}/trained_models/rtmscore_model1.pth"]
            self._run_cmd(" ".join(cmd))
        
        if not os.path.exists(rtm_out_fpath):
            return [99.99]
        else:
            with open(rtm_out_fpath) as lines:
                try:
                    scores = [-1. * float(x.split(",")[-1]) for x in lines if "id,score" not in x]
                except:
                    scores = [99.99,]
            return scores
        

if __name__ == "__main__":

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)

    sf = RtmscoreExtSF(receptor=receptor, ligand=ligand)
    print(sf.scoring())
    
