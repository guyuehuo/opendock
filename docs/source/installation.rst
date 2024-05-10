Installation 
============
Currently working on Linux. We did not test on Windows and Mac.

**OpenDock installation using pip**:

.. note::

    When using ``pip``, it's good pratice to use a virtual environment and also the easiest solution. An example with the `Conda package manager <https://docs.conda.io/en/latest/>`_ is available further down.
.. code-block:: bash
    
    $ pip install opendock #(not functional by now...)

**OpenDock installation in a Conda environment**:

The Anaconda Python distribution, which can be download from `https://docs.continuum.io/anaconda/install <https://docs.continuum.io/anaconda/install/>`_. This is a Python distribution specially designed for scientific applications, with many of the most popular scientific packages preinstalled. Alternatively, you can use `Miniconda <https://conda.pydata.org/miniconda.html>`_, which includes only Python itself, plus the Conda package manager.

1. Begin by installing the most recent 64 bit, Python 3.x version of either Anaconda or Miniconda
2. Create a dedicated environment for ``OpenDock``

.. code-block:: bash

    $ conda create -n opendock python=3
    $ conda activate opendock
    $ conda config --env --add channels conda-forge

3. Before you can use the framework, you may install the following python packages:
.. code-block:: bash   
    $ pip install numpy
    $ pip install pytorch 
    $ pip install pandas 

4. Install `OpenDock` package by using pip install:
.. code-block:: bash  
    $ cd opendock/
    $ pip install . 
