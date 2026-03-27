pytrist
=======

A Python reader for `Tristan-MP V2 <https://github.com/PrincetonUniversity/tristan-mp-v2>`_
plasma particle-in-cell (PIC) simulation output.

``pytrist`` loads simulation data from HDF5 output files and converts quantities
into physically meaningful **ion-based units** — ion inertial length, inverse ion
cyclotron frequency, and ion Alfvén speed — instead of the code's internal electron
units.

Installation
------------

.. code-block:: bash

   # From GitHub (recommended for HPCs):
   pip install git+https://github.com/colbych/pytrist.git

   # Development install:
   git clone https://github.com/colbych/pytrist.git
   cd pytrist && pip install -e ".[dev]"

**Requirements:** Python ≥ 3.9, numpy ≥ 1.21, h5py ≥ 3.0

Quick start
-----------

.. code-block:: python

   import pytrist

   sim = pytrist.Simulation("/path/to/output/")
   print(sim.steps)          # [1, 2, ..., 100]
   print(sim.summary())

   flds = sim.fields(step=10)
   bz   = flds.bz            # shape (nz, ny, nx)
   psi  = flds.psi()         # magnetic flux function

   fm   = sim.field_moments(step=10)
   vel  = fm.bulk_velocity(1, units="ion")   # {'x','y','z'} in vAi

   ef   = sim.energy_flux(step=10)
   S    = ef.poynting_flux(units="ion")      # {'x','y','z'} in n0 mi vAi³

.. toctree::
   :hidden:
   :maxdepth: 2

   api
