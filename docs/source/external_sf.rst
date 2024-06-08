.. _external_scoring_function:


Step-by-step external scoring function implementation
=====================================================

External scoring functions could be traditional or machine learning-based or deep learning-based scoring methods for protein-ligand binding prediction.
These scoring functions could be used as post-scoring methods for docking pose re-ranking after sampling (and clustering).

Some of the common scoring functions could be:

.. code-block:: bash

  OnionNet-SFCT
  RTMscore
  X-score
  zPoseScore

Or any other scoring functions that use the receptor and the docking pose as input. 
In the following part, we demostrate how external scoring functions are defined. 

1. OnionNet-SFCT
----------------------
OnionNet-SFCT is a machine learning-based scoring function that predicts the ligand docking pose RMSD (with regard to its native docking pose).
As a docking pose scoring function correction term, when it is combined with Vinascore (``VinaSF``), it is able to improve the docking success rate and virtual screening enrichment. 
The detail explanation of the scoring function could be found in this paper (https://doi.org/10.1093/bib/bbac051).  

                                                                             
                                                                            
