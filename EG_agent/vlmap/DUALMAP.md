## dualmap

This module provides open vocabulary mapping and generates navigation paths for robot navigation.

migrated and modified from DualMap:
![alt text](docs/dualmap.png)

- This module is the heaviest process of the agent, using ROS2 to obtain RGB-D and pose topics from the deployment env.
- This module should be upgradable with the advances of VLM cross-modal encoders

### Usage
Original detailed usage instructions and examples can refer to [DualMap](https://github.com/Eku127/DualMap?tab=readme-ov-file#applications).

outputs of this integrated module have been unified to the following single directory
```
EG_agent/vlmap/outputs/
└── carla
    ├── detections
    ├── detector_time.csv
    ├── log
    │   └── log_20250923_171446.log
    ├── map
    │   ├── 4421192e-711b-48e7-a3b9-6c18f03d7ed4.pkl
    │   ├── 55466905-c663-4f30-8696-267b34035f8e.pkl
    │   ├── 7154b243-e038-44ed-af25-de1a0984943a.pkl
    │   ├── a55db27c-aa64-46b8-b41c-4b090670906d.pkl
    │   ├── b85981be-5169-4b4e-8b0c-d568dcd9e45d.pkl
    │   ├── de4b00fd-12d1-4df5-a992-e8374b9d38e7.pkl
    │   └── layout.pcd
    └── system_time.csv
```

this module have been reorganized, simplified and wrapped for online interactive use, included in the agent system with the [interface](vlmap_nav_ros2.py)

#### Online Interactive Mapping
Before running, configure the [system_config.yaml](config/system_config.yaml):

```yaml
# Choose or create the appropriate class list depending on the scene:
# These classes would be used for yolov8l-world open-vocabulary zero-shot detection
given_classes_path: ./config/class_list/gpt_outdoor_general.txt
```

and [base_config.yaml](config/base_config.yaml)
```yaml
# set the rgbd sensor intrinsics
dataset_name: 'carla'
ros_topics:
  rgb: "/camera/rgb/image_raw"
  depth: "/camera/depth/image_raw"
  odom: "/camera/pose"
  camera_info: "/camera_info"
intrinsic:
  fx: 320
  fy: 320
  cx: 320
  cy: 240
# extrinsics:
#   [1, 0, 0, 0, 
#   0, 1, 0, 0, 
#   0, 0, 1, 0,
#   0, 0, 0, 1]
```

### Note
to be considered earlier, may extend support for UAV bird-eye view mappings, where the depth should be replaced with height or other representation.