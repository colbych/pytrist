API Reference
=============

All classes are importable from the top-level ``pytrist`` namespace or from
their individual modules.

.. contents:: Classes
   :local:
   :depth: 1

----

Primary entry point
-------------------

.. autoclass:: pytrist.Simulation
   :members:
   :member-order: bysource

----

Data snapshots
--------------

.. autoclass:: pytrist.fields.FieldSnapshot
   :members:
   :member-order: bysource

.. autoclass:: pytrist.particles.ParticleSnapshot
   :members:
   :member-order: bysource

.. autoclass:: pytrist.spectra.SpectraSnapshot
   :members:
   :member-order: bysource

.. autoclass:: pytrist.history.History
   :members:
   :member-order: bysource

.. autoclass:: pytrist.params.SimParams
   :members:
   :member-order: bysource

----

Derived diagnostics
-------------------

.. autoclass:: pytrist.FieldMoments
   :members:
   :member-order: bysource

.. autoclass:: pytrist.moments.ParticleMoments
   :members:
   :member-order: bysource

.. autoclass:: pytrist.EnergyFlux
   :members:
   :member-order: bysource

----

Unit conversion
---------------

.. autoclass:: pytrist.UnitConverter
   :members:
   :member-order: bysource
