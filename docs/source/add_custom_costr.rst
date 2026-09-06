.. add_custom_constraint:


Step-by-step custom constraints implementation
=====================================================

Users can customize various types of constraints, such as distance constraints, angle constraints, dihedral angle constraints, hydrogen bond constraints, etc

Some of the common scoring functions could be:

.. code-block:: bash

  distance constraints
  angle constraints
  hydrogen bond constraints

The current framework implements distance constraints.
In order to guide users to implement custom constraints, taking the example of adding angle constraints,
in the following part,we will demonstrate in detail how to implement angle constraints
 

1. Define angle constraint class--AngleConstraintSF
---------------------------------------------------
The added custom constraints can be understood as a new scoring function, 
so first go to the scorer folder and open constraints.py

.. code-block:: bash
    
    vim opendock/scorer/constraints.py

In the constrains.py file, it can be seen that distance constraints have been implemented, 
such as DistanceConstraintSF and DistanceMatrixConstraintSF.
To add angle constraints, the first step is to implement an 
angle calculation function located in the base class ConstraintSF.
The input is 3 coordinate points, and for ease of calculation, the output is the radian value of the angle.

.. code-block:: bash

    def _angle(self, x, y, z):
        # Calculate the two vectors
        vec1 = x - y
        vec2 = z - y
        
        # Compute the dot product of the two vectors
        dot_product = torch.sum(vec1 * vec2)
        
        # Compute the magnitudes of the vectors
        mag1 = torch.sqrt(torch.sum(torch.pow(vec1, 2)))
        mag2 = torch.sqrt(torch.sum(torch.pow(vec2, 2)))
        
        # Calculate the cosine of the angle using the dot product formula
        cos_theta = dot_product / (mag1 * mag2)
        
        # Ensure cos_theta is within the valid range [-1, 1] to avoid numerical errors
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Calculate the angle (in radians) using the arccosine function
        angle = torch.acos(cos_theta)
        
        return angle

Then we can define AngleConstraintSF,
which inherits from the base class ConstraintSF

.. code-block:: bash

    class AngleConstraintSF(ConstraintSF):

      def __init__(self, 
                 receptor = None,
                 ligand = None, 
                 **kwargs):
        super(AngleConstraintSF, self)\
        .__init__(receptor=receptor, ligand=ligand)

        self.grpA_mol_ = kwargs.pop('groupA_mol', "receptor") 

        self.grpB_mol_ = kwargs.pop('groupB_mol', "receptor") # receptor or ligand

        self.grpC_mol_ = kwargs.pop('groupC_mol', "ligand")  

        self.grpA_idx_ = kwargs['grpA_ha_indices']
        self.grpB_idx_ = kwargs['grpB_ha_indices']
        self.grpC_idx_ = kwargs['grpC_ha_indices']

        self.constraint_type_ = kwargs.pop('constraint', 'harmonic')
        self.force_constant_ = kwargs.pop('force', 1.0)
        #self.constraint_reference_ = kwargs.pop('reference', None)
        # angle boundary, unit is angstrom
        self.bounds_ = kwargs.pop('bounds', [0, np.pi])


        assert (len(self.grpA_idx_) > 0 and len(self.grpB_idx_) > 0 and len(self.grpC_idx_) > 0)

      def scoring(self):

        if self.grpA_mol_.lower() in ['receptor', 'protein']:
            _grpA_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpA_mol_.lower() in ['ligand', 'molecule']:
            _grpA_xyz = self.ligand.pose_heavy_atoms_coords[0]
        
    
        if self.grpB_mol_.lower() in ['receptor', 'protein']:
            _grpB_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpB_mol_.lower() in ['ligand', 'molecule']:
            _grpB_xyz = self.ligand.pose_heavy_atoms_coords[0]

        if self.grpC_mol_.lower() in ['receptor', 'protein']:
            _grpC_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpC_mol_.lower() in ['ligand', 'molecule']:
            _grpC_xyz = self.ligand.pose_heavy_atoms_coords[0]
      
        pairs = list(zip(self.grpA_idx_, self.grpB_idx_,self.grpC_idx_))
     
        self._angle_paired_ = []
        for i, (atm1, atm2,atm3) in enumerate(pairs):
            _a = self._angle(_grpA_xyz[atm1], _grpB_xyz[atm2], _grpC_xyz[atm3])
            self._angle_paired_.append(_a)

        self.angle_paired = torch.stack(self._angle_paired_)
     
        score = self._apply_constraint(torch.mean(self.angle_paired))

        return score.reshape((1, -1))


Thus, we successfully defined the angle constraint through the class AngleConstrainSF. 
Next, we will demonstrate how to use the newly defined angle constraint

2. Use newly defined angle constraints
----------------------------------------

Consistent with previous constraint cases, taking pdb: 3gzj as an example, we chose ligand atom CAF,
protein side chain atoms ser87-OG, and GLY19-O as cases

.. note::
    Please note that the residue index (``residx``) is generally 1-based as indicated in the PDB file.
    The above atomic names have some differences between protein PDB files and PDBQT files, but the atoms are the same.

.. code-block:: bash
    
    vim opendock/example/3gzj/atom_angle_constraint_example.py
   
.. code-block:: bash

    indices_r1 = asl.select_atom(atomnames=['O', ], chains=['A'], residx=['10'], resnames=['GLY'])
    print("indices_r1",indices_r1)
    print(indices_r1, receptor.dataframe_ha_.head())

    indices_r2 = asl.select_atom(atomnames=['OG', ], chains=['A'], residx=['78'], resnames=['SER'])
    asl = AtomSelection(molecule=ligand)
   
    indices_l = asl.select_atom(atomnames=['CAF', ])
 
    cnstr = AngleConstraintSF(receptor, ligand,
                                 grpA_ha_indices=indices_r1,
                                 grpB_ha_indices=indices_r2,
                                 grpC_ha_indices=indices_l,
                                 constraint='wall',
                                 bounds=[1.0698, 1.0698]  #1.0698 indicate 61.3°, choose your custom angle upper and lower limits 
                                 )
    print(cnstr.scoring())

    # vina scoring function
    sf1 = VinaSF(receptor, ligand)
    vs = sf1.scoring()
    print("Vina Score ", vs)

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[sf1, cnstr], weights=[0.5, 0.5])
    vs = sf.scoring()
    print("HybridSF Score ", vs)

So far, we have successfully defined the angle constraint and used it. 
Let's develop your own custom constraints!
                                  
