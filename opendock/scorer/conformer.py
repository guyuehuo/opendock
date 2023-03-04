

from opendock.scorer.scoring_function import BaseScoringFunction
from opendock.scorer.vina import VinaSF


class ConformerVinaIntraSF(VinaSF):
    """
    Calculate the intra-molecular energy of a ligand given its conformations.

    Methods
    -------
    scoring: torch.Tensor, shape = (-1, 1)

    Attributes
    ----------
    ligand: opendock.core.conformation.LigandConformation
        The ligand object. 
    receptor: opendock.core.conformation.ReceptorConformation
        The receptor object. It is not required for ligand 
        conformer generation.
    """

    def __init__(self, ligand = None, receptor = None, **kwargs):
        super(ConformerVinaIntraSF, self).__init__(ligand=ligand, 
                                                   receptor=receptor)
    
    def scoring(self):
        """Calculate the ligand intra-molecular energy using 
        AutoDock Vina intra-term. 

        Returns
        ------- 
        score: torch.Tensor, shape = (1, 1)
            The returned energy of the ligand.
        """

        self.ligand.cnfr2xyz(self.ligand.cnfrs_)

        return self.cal_intra_repulsion()
    