# Define object/category sets for embodied actions

AllObject = {
    "StagingArea", "Corridor", "Intersection", "Stairwell",
    "Office", "Warehouse", "ControlRoom", "LoadingBay",
    "Lobby", "ChargingStation", "Outdoor", "Indoor", "Wall", 
    "Doorway", "Window", "ElectricalPanel", "GasMeter", "Equipment",
    "StructuralCrack", "SmokeSource", "WaterLeak", "BlockedExit",
    "Person", "Car", "Truck", "Bicycle", "Motorcycle",
    "Blood", "Fire", "Gas", "Debris", 
    "Victim", "Rescuer", "Visitor", "Staff"
}

AllCondition = 'RobotNear_<AllObject>, IsCaptured_<AllObject>, IsMarked_<AllObject>, IsReported_<AllObject>'
