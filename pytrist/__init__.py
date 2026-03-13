"""
pytrist — Python reader for Tristan-MP V2 plasma PIC simulation output.

Primary entry point::

    import pytrist

    sim = pytrist.Simulation("/path/to/simulation/output/")
    print(sim.steps)

    flds = sim.fields(step=10)
    bx   = flds.bx

    prtl = sim.particles(step=10)
    sp1  = prtl.species(1)

    hist = sim.history()
    t    = hist["t"]

    uc   = sim.unit_converter
    print(uc.summary())

Classes can also be imported directly::

    from pytrist import Simulation, UnitConverter, SimParams
    from pytrist.fields import FieldSnapshot
    from pytrist.particles import ParticleSnapshot
    from pytrist.history import History
    from pytrist.spectra import SpectraSnapshot
"""

from .simulation import Simulation
from .fields import FieldSnapshot, FieldLoader
from .particles import ParticleSnapshot
from .moments import ParticleMoments
from .params import SimParams
from .history import History
from .spectra import SpectraSnapshot
from .units import UnitConverter

__version__ = "0.1.0"
__author__ = "pytrist contributors"

__all__ = [
    "Simulation",
    "FieldSnapshot",
    "FieldLoader",
    "ParticleSnapshot",
    "ParticleMoments",
    "SimParams",
    "History",
    "SpectraSnapshot",
    "UnitConverter",
    "__version__",
]
