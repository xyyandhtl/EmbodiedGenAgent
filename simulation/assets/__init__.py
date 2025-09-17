"""Package containing asset and sensor configurations."""

import os

##
# Configuration for different assets.
##

# Conveniences to other module directories via relative paths
SIMULATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the simulation source directory."""

SIMULATION_DATA_DIR = os.path.join(SIMULATION_DIR, "data")
"""Path to the simulation data directory."""