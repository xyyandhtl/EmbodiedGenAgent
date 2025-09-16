# Define reusable object/category sets for embodied actions

LOCATIONS = {
    "StagingArea", "Corridor", "Intersection", "Stairwell",
    "Office", "Warehouse", "ControlRoom", "LoadingBay",
    "Lobby", "ChargingStation", "Outdoor", "Indoor", "Wall"
}

INSPECTION_POINTS = {
    "Doorway", "Window", "ElectricalPanel", "GasMeter", "Equipment",
    "StructuralCrack", "SmokeSource", "WaterLeak", "BlockedExit",
}

INCIDENTS = {
    "Blood", "Fire", "Gas", "Debris"
}

PERSONS = {
    "Victim", "Rescuer", "Visitor", "Staff"
}

# Derived sets for convenience
NAV_POINTS = LOCATIONS | INSPECTION_POINTS
CAPTUREABLE = INSPECTION_POINTS | INCIDENTS | PERSONS
MARKABLE = CAPTUREABLE
REPORTABLE = CAPTUREABLE

# Backward-compat: single roll-up set if needed elsewhere
AllObject = NAV_POINTS | CAPTUREABLE

__all__ = [
    "LOCATIONS",
    "INSPECTION_POINTS",
    "INCIDENTS",
    "PERSONS",
    "NAV_POINTS",
    "CAPTUREABLE",
    "MARKABLE",
    "REPORTABLE",
    "AllObject",
]
