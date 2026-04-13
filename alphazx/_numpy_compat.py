"""
Compatibility shim for NumPy 2.0+.

NumPy 2.0 removed several deprecated type aliases (np.float_, np.int_,
np.complex_, np.object_, np.bool_).  networkx 3.4.x still references
np.float_ and np.int_ in its GraphML reader, which is on the critical
path for pyzx graph serialization.

This module restores the removed aliases so that networkx (and any other
library relying on them) continues to work until upstream fixes land.

Import this module as early as possible — before networkx or pyzx.
"""

import numpy as np

_PATCHED_ATTRS = {
    "float_": np.float64,
    "int_": np.int64,
    "complex_": np.complex128,
    "object_": np.object_,
    "bool_": np.bool_,
}

for _attr, _fallback in _PATCHED_ATTRS.items():
    if not hasattr(np, _attr):
        setattr(np, _attr, _fallback)
