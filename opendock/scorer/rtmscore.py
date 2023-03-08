import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda
#sys.path.append("/home/shenchao/resdocktest2/rtmscore2")
sys.path.append(os.path.abspath(__file__).replace("rtmscore.py",".."))
from torch.utils.data import DataLoader
from opendock.scorer.RTMScore.RTMScore.data.data import VSDataset
from opendock.scorer.RTMScore.RTMScore.model.utils import collate, run_an_eval_epoch
from opendock.scorer.RTMScore.RTMScore.model.model2 import RTMScore, DGLGraphTransformer #LigandNet, TargetNet
import torch.multiprocessing
from opendock.scorer.RTMScore.utils import obabel
torch.multiprocessing.set_sharing_strategy('file_system')

_current_dpath = os.path.dirname(os.path.abspath(__file__))
RTMScore_Model = os.path.join(_current_dpath, "RTMScore/trained_models/rtmscore_model1.pth")

args = {}
args["batch_size"] = 128
args["dist_threhold"] = 5
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args['device'] = "cpu"
args["num_workers"] = 10
args["num_node_featsp"] = 41
args["num_node_featsl"] = 41
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.10

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
    # try:
    if not os.path.exists(".rtm_temp"):
        os.mkdir(".rtm_temp")

    if not prot.endswith("pdb"):
        out = ".rtm_temp/" + os.path.basename(prot).split(".")[0] + ".pdb"
        obabel(prot, out)
        prot = out

    if not lig.endswith("mol2"):
        out = ".rtm_temp/" + os.path.basename(lig).split(".")[0] + ".mol2"
        obabel(lig, out)
        lig = out

    if not reflig.endswith("mol2"):
        out = ".rtm_temp/" + os.path.basename(reflig).split(".")[0] + ".mol2"
        obabel(reflig, out)
        reflig = out

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

    ids = data.ids
    df = pd.DataFrame(zip(*(ids, preds)), columns=["id", "score"])
    df.sort_values("score", ascending=False, inplace=True)
    print(df)

    return data.ids, preds

if __name__ == "__main__":

    os.mkdir(".rtm_temp")

    lig_fpath = sys.argv[1]  # poses
    rec_fpath = sys.argv[2]
    ref_lig_fpath = sys.argv[3]  # the reference ligand (native conformation)

    ids, scores = rtmsf(prot=rec_fpath,
                          lig=lig_fpath,
                          modpath=RTMScore_Model,
                          cut=10.0,
                          gen_pocket=True,
                          reflig=ref_lig_fpath,
                          explicit_H=False,
                          use_chirality=True,
                          parallel=False,
                          **args
                          )
    
