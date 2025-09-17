## dualmap

This module provides open vocabulary mapping and generates navigation paths for robot navigation.

migrated and modified from DualMap:
![alt text](docs/dualmap.png)

### Usage
Original detailed usage instructions and examples can refer to [DualMap](https://github.com/Eku127/DualMap?tab=readme-ov-file#applications).

#### Online Interactive Mapping
Before running, configure the following YAML file:

[system_config.yaml](config/system_config.yaml)
```yaml
# Choose or create the appropriate class list depending on the scene:
# These classes would be used for yolov8l-world open-vocabulary zero-shot detection
given_classes_path: ./config/class_list/gpt_outdoor_general.txt
```

[base_config.yaml](config/base_config.yaml)
```yaml
# set the rgb sensor intrinsics
camera_params:
  image_height: 480
  image_width: 640
  fx: 480.0
  fy: 480.0
  cx: 320.0
  cy: 240.0
```

### Note
to be considered earlier, may extend support for UAV bird-eye view mappings, where the depth should be replaced with height or other representation.