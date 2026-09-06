.. _Protein side chain optimization:

Protein side chain optimization
===============================

Besides ligand conformation optimization, Opendock can also optimize protein side chains, 
such as selecting specific protein side chains or all side chains that are in close proximity to the ligand conformation.

The following is an example of protein side-chain optimization, where the conformation of the ligand remains fixed,
and side chains of the protein within a distance of 8 Å from the ligand are optimized.

.. code-block:: bash
    
    from opendock.scorer.vina import VinaSF
    #Initialize protein side chains
    sc_list = receptor.init_sidechain_cnfrs()
    #Output the name of the selected side chain for optimization
    print("SC_CNFR_LIST", sc_list, receptor.selected_residues_names)
    sc_cnfrs = torch.cat(sc_list)
    print(sc_cnfrs)
    #Convert side chain vectors into coordinates
    print(receptor.cnfr2xyz(sc_list))
    #Set the ligand vector to none, keeping the ligand conformation unchanged
    ligand.init_cnfrs=None
    
    #Output side chain vector
    init_recp_cnfrs = sc_list
    print("init_recp_cnfrs",init_recp_cnfrs)
    #Conduct sampling
    .......

For this tutorial, all the basic material are provided and can be found 
in the ``opendock/opendock/protocol`` directory

You can find this script in the ``example`` folder of OpenDock available on Github. To execute it from a command line,
go to your terminal/console/command prompt window. Navigate to the ``examples`` folder by typing

.. code-block:: console

    $ cd opendock/example/1gpn
    $ python side_chain_optimization_example.py -c vina.config # Protein side chain optimization.
   