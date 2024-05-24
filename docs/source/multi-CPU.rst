.. _Multi-CPU parallel execution:

Multi-CPU parallel execution
============================
For all multi CPU parallel docking files, they can be found in the ``opendock/opendock/protocol`` directory.
For some systems, there may be system resource limitations when performing multi CPU parallelism,
so another multi CPU parallelism approach is provided in ``another-way``,
you can enter the corresponding directory by typing the following command:

.. code-block:: console

    $ cd opendock/protocol/redocking/another-way      #enter the redocking protocol directory
    $ cd opendock/protocol/crossdocking/another-way   #enter the crossdocking protocol directory
    $ cd opendock/protocol/constrain                  #enter the custom constraint protocol directory