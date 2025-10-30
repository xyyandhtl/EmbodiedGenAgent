def setup_isaacsim_settings():
    """Setup Isaac Sim settings to hide the UI."""
    import carb.settings
    # print(f'omni appwindow: {omni.appwindow.__file__}')
    # Use SimulaitonApp directly should set carb_settings
    carb_settings_iface = carb.settings.get_settings()
    # carb_settings_iface.set("/persistent/isaac/asset_root/cloud", 
    #                         "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0")
    carb_settings_iface.set("/app/window/maximized", False)
    carb_settings_iface.set("/app/window/fullscreen", True)
    # carb_settings_iface.set("/app/window/width", 1280)
    # carb_settings_iface.set("/app/window/height", 720)
    carb_settings_iface.set("/app/window/hideUi", True)
    # carb_settings_iface.set("/isaaclab/cameras_enabled", True)
        
def hide_workspace_windows():
    """Hide all workspace windows in Isaac Sim."""
    from omni.ui import Workspace
    Workspace.show_window("Stage", False)
    Workspace.show_window("Layer", False)
    Workspace.show_window("Property", False)
    Workspace.show_window("Render Settings", False)
    Workspace.show_window("Console", False)
    Workspace.show_window("Content", False)

def get_current_stage():
    """Get the current USD stage."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    # terrain_prim = stage.GetPrimAtPath("/World/Terrain")
    return stage