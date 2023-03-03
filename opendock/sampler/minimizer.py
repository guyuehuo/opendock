from torch.optim import Adam, LBFGS, SGD
import torch
import os, sys


def sgd_minimizer(x, target_function, **kwargs):
    # Define the optimizer
    nsteps=kwargs.pop('nsteps', 20)
    lr    = kwargs.pop('lr', 0.1)
    
    optimizer = SGD(x, lr=lr, weight_decay=0.01, momentum=0.9)

    for i in range(nsteps):
        optimizer.zero_grad()
        loss = target_function(x)
        loss.backward(retain_graph=True)
        optimizer.step()

    return x


def adam_minimizer(x, target_function, **kwargs):
    # Define the optimizer
    nsteps=kwargs.pop('nsteps', 20)
    lr    = kwargs.pop('lr', 0.1) 

    optimizer = Adam(x, lr=lr, weight_decay=0.01)

    for i in range(nsteps):
        optimizer.zero_grad()
        loss = target_function(x)
        loss.backward(retain_graph=True)
        optimizer.step()

    return x

def lbfgs_minimizer(x, target_function, **kwargs):

    # Define the optimizer
    optimizer = LBFGS(x, lr=0.1, history_size=5, max_iter=10)

    # Add the closure function to calculate the gradient.
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        
        loss = target_function(x)

        if loss.requires_grad:
            loss.backward(retain_graph=True)
        return loss
    
    # optimize now
    optimizer.step(closure)

    return x


if __name__ == "__main__":
    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.core import io

    ligand = LigandConformation(sys.argv[1])
    #ligand.parse_ligand()
    #print("ligand.init_heavy_atoms_coords", ligand.init_heavy_atoms_coords)

    receptor = ReceptorConformation(sys.argv[2], ligand.init_heavy_atoms_coords)
    #print("InitFirstAtom", receptor.init_rec_heavy_atoms_xyz[0])
    #receptor.parse_receptor()
    #print(receptor.init_rec_heavy_atoms_xyz, receptor.init_rec_heavy_atoms_xyz.shape)

    # cnfr
    cnfr = ligand.init_cnfrs[0]
    xyz_init = ligand.init_heavy_atoms_coords
    # sc cnfr
    sc_list = receptor.init_sidechain_cnfrs()
    #sc_cnfrs = torch.cat(sc_list)
    X = [cnfr, ] + sc_list
    cnfr_list = [cnfr, ]
    scores = [0,]

    sf = DeepRmsdSF(receptor, ligand)
    _dist1 = sf.generate_pldist_mtrx()

    rec_cnfr_list = []

    def target_function_1(x):
        # x is a list of Tensors
        return torch.pow((x[0][0] + x[0][1]), 2)

    def sf_2(x):
        # given the cnfr, redefine pose_xyz
        #print("Init x", x[0])
        # define the pose cnfr
        #ligand.cnfr2xyz(x[0])

        # receptor sidechain 
        #print("Receptor sidechain", x)
        _xyz = receptor.cnfr2xyz(x)
        rec_cnfr_list.append([torch.Tensor(y.detach().numpy()) for y in x])
        #print("OptimDiff", [sc_list[_x] - x[_x] for _x in range(len(x))])
        #print("TargetFirstAtom", receptor.rec_heavy_atoms_xyz[0])
        #receptor.rec_heavy_atoms_xyz = _xyz * 1.0

        delta = torch.sum(receptor.init_rec_heavy_atoms_xyz - receptor.rec_heavy_atoms_xyz, axis=1)
        #print("ReceptorSidechainShift", delta.sum())

        # get ligand center
        print("CENTER", ligand._get_geo_center())
        # calculate rmsd
        rmsd = rmsd_to_reference(ligand.pose_heavy_atoms_coords, xyz_init) 
        print("RMSD ", rmsd)       

        # vina energy    
        sf = VinaSF(receptor, ligand)
        sf.rec_heavy_atoms_xyz = _xyz * 1.0
        _dist2 = sf.generate_pldist_mtrx()

        distsum = (_dist1 - _dist2).sum()
        print("DistanceMatrixDiff ", distsum)

        ds = torch.sum(sf.scoring())
        #vscore = sf.scoring()
        print("DeepRMSD ", ds)

        # cnfr list
        #cnfr_list.append(torch.Tensor((x[0] * 1.0).detach().numpy()))
        scores.append(ds)

        return ds

    nx = sgd_minimizer(sc_list, sf_2)
    print(nx)
    
    # write trajectory
    #io.write_receptor_traj(rec_cnfr_list, receptor, sys.argv[3])
