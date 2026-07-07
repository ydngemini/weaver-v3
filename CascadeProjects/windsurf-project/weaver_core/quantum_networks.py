"""
Compatibility import for Weaver's Kingston manifold quantum layer.

The source of truth lives at the project root in quantum_networks.py so the
standalone quantum loop and older scripts import the same implementation.
"""

from quantum_networks import *  # noqa: F401,F403
