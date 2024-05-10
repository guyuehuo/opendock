.. _basic_docking:

Basic docking
=============

Let's start with our first example of docking, where the typical usage pattern would be to dock a single molecule into a rigid receptor.
In this example, Monte Carlo sampling(``MonteCarloSampler``) will be used, and the scoring function will employ Vinascore (``VinaSF``).
In this example we will dock the PDB entry 1gpn using OpenDock.

1. Preparing the receptor and ligand
-------------------------

During this step, we will create a PDBQT file of our receptor containing only the polar hydrogen atoms as well as partial charges.
Conversion can be done using OpenBabel.

.. code-block:: bash

    $ obabel 1gpn_receptor.sdf -o 1gpn_receptor.pdbqt

.. note::

    If you have not installed openbabel, you can install it using the following command

.. code-block:: bash
    
    $ pip install openbabel

2. Preparing the ligand
-----------------------

This step is very similar to the previous step. We will also create a PDBQT file from a ligand molecule file (in MOL/MOL2 or SDF format)

.. warning::
  
  We strongly advice you against using PDB format for preparing small molecules, since it does not contain information about bond connections. 
  Please don't forget to always check the protonation state of your molecules before docking. Your success can sometimes hang by just an hydrogen atom.

.. code-block:: bash

    $ obabel 1gpn_ligand.sdf -o 1gpn_ligand.pdbqt

3. Prepare configuration files

You can generate a configuration file by running the prepare_configs.py file,which is located in ``opendock/opendock/test
/Prepare_configs.py``
