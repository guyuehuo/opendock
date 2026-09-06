from torch.optim import Adam, LBFGS, SGD
import torch.optim as optim
from opendock.sampler.base import BaseSampler
import torch
import os, sys
import math
import random
from scipy.optimize import minimize

def cg_minimizer(x, target_function, **kwargs):
    # Define the optimizer
    
    nsteps = kwargs.pop('nsteps', 20)
    lr = kwargs.pop('lr', 0.1)
    x_np = x[0].detach().numpy()  
   
    def loss_function(x_np):
        
        x = [torch.tensor([x_np],dtype=torch.float32, requires_grad=True)]  # Convert the NumPy array back to a PyTorch tensor
        #x=[torch.Tensor([x_np]).requires_grad()]

       
        loss = target_function(x)
       
        loss.backward(retain_graph=True)
        return loss.item()

  
    #result = minimize(loss_function, x_np, method='CG', options={'maxiter': nsteps, 'gtol': lr})
    result = minimize(loss_function, x_np, method='CG', options={'maxiter':20})
   
    x =[torch.tensor([result.x], dtype=torch.float32,requires_grad=True)]  # 
   
    return x

def cg_minimizer1(x, target_function, **kwargs):
    # Define the optimizer
    nsteps = kwargs.pop('nsteps', 10)
    lr = kwargs.pop('lr', 0.1)

    # Initialize the optimizer with Conjugate Gradient (CG) algorithm
    optimizer = optim.ConjugateGradient([x], lr=lr)


    for i in range(nsteps):
        optimizer.zero_grad()
        loss = target_function(x)
        loss.backward(retain_graph=True)

        # Optimize using CG optimizer
        try:
            optimizer.step()
        except:
            print("[WARNING] minimization failed, skip it...")

    return x

def sgd_minimizer(x, target_function, **kwargs):
    # Define the optimizer
    nsteps=kwargs.pop('nsteps', 20)
    lr= kwargs.pop('lr', 0.2)
    
    optimizer = SGD(x, lr=lr, weight_decay=0.1, momentum=0.8)

    for i in range(nsteps):
        optimizer.zero_grad()
        loss = target_function(x)
        #print('loss',loss)
        #print('x',x)
        loss.backward(retain_graph=True)
        # optimize now
        try:
            optimizer.step()
        except:
            print("[WARNING] minimization failed, skip it...")

    return x


def adam_minimizer(x, target_function, **kwargs):
    # Define the optimizer
    nsteps=kwargs.pop('nsteps', 20)
    lr    = kwargs.pop('lr', 0.1)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #x=x[0].to(device).detach()
    #x.requires_grad = True
    #x = x.to(device)
    #x=[x]
    #print(x)
    optimizer = Adam(x, lr=lr, weight_decay=0.01)
    #print('success1')
    for i in range(nsteps):
        optimizer.zero_grad()
        #print("adam",x)
        loss = target_function(x)
        #print('loss', loss)
        loss.backward(retain_graph=True)

        # optimize now
        try:
            optimizer.step()
        except:
            print("[WARNING] minimization failed, skip it...")

    return x


def lbfgs_minimizer(x, target_function, **kwargs):
    # Define the optimizer
    nsteps = kwargs.pop('nsteps', 5)
    lr = kwargs.pop('lr', 0.05)

    if nsteps <= 2:
        nsteps = 2

    # Define the optimizer
    optimizer = LBFGS(x, lr=lr, history_size=nsteps, max_iter=nsteps)

    # Add the closure function to calculate the gradient.
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        loss = target_function(x)
        #print('loss:', loss)
        #print('cnfrs:', x)
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    # optimize now
    # try:
    #     optimizer.step(closure)
    # except:
    #     print("[WARNING] minimization failed, skip it...")

    return x

def lbfgs_minimizer1(x, target_function, **kwargs):
   
    # Define the optimizer
    nsteps = kwargs.pop('nsteps', 5)
    lr = kwargs.pop('lr', 0.05)

    if nsteps <= 2:
        nsteps = 2

    # Move the variables to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.is_available():
   
    #else:
   
    #x1=x[0]
    #x= x.to(device)
    
    # Define the optimizer
    # optimizer = LBFGS(x, lr=lr, history_size=nsteps, max_iter=nsteps)
    # #print('success1')
    # # Add the closure function to calculate the gradient.
    # def closure():
    #     if torch.is_grad_enabled():
    #         optimizer.zero_grad()
    #
    #     loss = target_function(x)
    #
    #     if loss.requires_grad:
    #         loss.backward(retain_graph=True)
    #     return loss
    #
    # #print('success2')
    # # Optimize now
    # try:
    #     optimizer.step(closure)
    #     #print('success3')
    # except:
    #     print("[WARNING] Minimization failed, skip it...")
    #
    # # Move the variables back to CPU if necessary
    # #x = x.to("cpu")
    #
    # return x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x[0].to(device)
    #print(x)
    x=x.detach()
    x.requires_grad=True
    #x=[x]
    # x_gpu = x.tolist()
    # x_gpu = torch.tensor(x_gpu)
    # x_gpu = x_gpu.to(device)
    # x_gpu.requires_grad_(x.requires_grad)
    # x_gpu = [x_gpu]
    # x= x.to(device)
  
    # Define the optimizer
    optimizer = LBFGS([x], lr=lr, history_size=nsteps, max_iter=nsteps)
    #print('success1')

    # Add the closure function to calculate the gradient.
    def closure():
        #if torch.is_grad_enabled():
        optimizer.zero_grad()
        #print('ss1')
        y=x.to("cpu")
        #print(x)
        loss = target_function([y])
        #print(loss)
        #print('ss2')
        #if loss.requires_grad:
        loss.backward(retain_graph=True)
        #print('ss3')
        return loss

    #print('success2')
    # Optimize now
    try:
      optimizer.step(closure)
    #print('success3')
    except:
      print("[WARNING] Minimization failed, skip it...")

    # Move the variables back to CPU if necessary
    x = x.to("cpu")
   
    return [x]


class MinimizerSampler(BaseSampler):
    def __init__(self, receptor, ligand, scoring_function, **kwargs):
        super(MinimizerSampler, self).__init__(receptor, ligand, scoring_function)
        self.receptor = receptor
        self.ligand = ligand

        self.lr_mode = kwargs.pop('lr_mode', 'periodic')
        #self.method = kwargs.pop('method', 'Adam')
        self.minimizer = kwargs.pop('minimizer', None)
        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)
    
    def _make_periodic_lr(self, total_step=1000, 
                          current_step=0, 
                          init_lr=1.0, rounds=10):
        """Obtain periodic learning rate for a given step.
        
        Args
        ---- 
        total_step: int, 
            number of total sampling steps
        current_step: int, 
            current step index 
        init_lr: float, optional, default = 1.0
            initial learning rate for the sampling 
        rounds: int, optional, default = 10
            number of rounds for learning rate rising

        Returns
        -------
        lr: float, 
            the learning rate for minimization
        """
        chunck = int(total_step / rounds) 
        ratio = ((current_step % chunck) / chunck)

        #return (2 * init_lr * abs(0.5 - ratio)) ** 2 + 1e-4
        return init_lr * (1 - ratio) + 1e-4

    def sampling(self, nsteps=1000, 
                 init_lr=1.0, 
                 rounds=10):

        for _step in range(nsteps):
            self.index_ = _step
            _lr = self._make_periodic_lr(total_step=nsteps, 
                                         current_step=_step, 
                                         init_lr=init_lr,
                                         rounds=rounds)
            #print("LR ", _lr)
            if self.ligand.cnfrs_ is not None:
                is_ligand = True 
            else: 
                is_ligand = False
            
            if self.receptor.cnfrs_ is not None:
                is_receptor = True
            else:
                is_receptor = False

            self.ligand.cnfrs_, self.receptor.cnfrs_ = \
            self._minimize(self.ligand.cnfrs_, self.receptor.cnfrs_, 
                           is_ligand=is_ligand, is_receptor=is_receptor,
                           lr=_lr, nsteps=1,
                           )
            if self.ligand.cnfrs_ is not None:
                #print(self.ligand.cnfrs_[0])
                self.ligand_cnfrs_history_.append(torch.Tensor(self.ligand.cnfrs_[0]\
                                                               .detach().numpy() * 1.0))
                #print(self.ligand_cnfrs_history_[-1])
            if self.receptor.cnfrs_ is not None:
                self.receptor_cnfrs_history_.append([torch.Tensor(x.detach().numpy() * 1.0) \
                                                     for x in self.receptor.cnfrs_])
                
            _score = self._score(self.ligand.cnfrs_, self.receptor.cnfrs_)
            _score = _score.detach().numpy().ravel()[0]
            self.ligand_scores_history_.append(_score)

            print(f"[INFO] #{self.index_} {self.__class__.__name__} score {_score}")
        
        return self
    

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
