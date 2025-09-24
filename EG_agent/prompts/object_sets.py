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
TARGETS = INSPECTION_POINTS | INCIDENTS | PERSONS

CAPTUREABLE = TARGETS
MARKABLE = TARGETS
REPORTABLE = TARGETS

# Backward-compat: single roll-up set if needed elsewhere
AllObject = NAV_POINTS | TARGETS

AllCondition = 'RobotNear_<NAV_POINTS>, IsCaptured_<TARGETS>, IsMarked_<TARGETS>, IsReported_<TARGETS>'

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
