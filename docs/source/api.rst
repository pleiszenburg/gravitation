:github_url:

.. _api:

API
===

Inventory
---------

*gravitation* exposes all available kernels through a dictionary-like object, `inventory`. Initially, `inventory` only provides a "list" of available kernels. Kernel meta data must be loaded manually (`load_meta`). The kernel's Python (sub-) module also must be imported manually (`load_module`). Meta data is loaded without importing the kernel.

Proceedure
----------

Kernels have to be "started" before they can perform any type of computation (`start`). Once they are started, they can compute as many time steps as desired (`step`). If a kernel object is supposed to be discarded, it can be "stopped" (`stop`). A stopped kernel can not be used for computations. Bodies / point masses must be added to a kernel (`add_object`) before it is started.

.. code:: python

    from gravitation import inventory

    inventory[kernel_name].load_meta() # loads meta data from kernel source
    fields = inventory[kernel_name].keys() # provides access to kernel meta data field names
    meta = inventory[kernel_name][field_name] # provides access to kernel meta data

    inventory[kernel_name].load_module() # attempts to import kernel module
    Kernel = inventory[kernel_name].get_class() # returns kernel's universe class
    kernel = inventory[kernel_name](*args, **kwargs) # returns instance of kernel's universe class

    kernel.add_mass(pm) # adds point mass to universe
    kernel.start() # runs initialization routine(s) for kernel prior computations
    kernel.step() # computes one time step
    kernel.stop() # runs clean-up routine(s) after a kernel has been used

.. toctree::
    :maxdepth: 2
    :caption: Infrastructure

    universebase
    pointmass
