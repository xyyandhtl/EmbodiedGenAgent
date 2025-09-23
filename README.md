# EmbodiedGenAgent
A Generative Embodied Agent that transforms open-ended instructions into behavior trees for autonomous decision-making and action execution loop


## Plans
- [x] Implement the intent understanding: vlm prompt / reflective feedback to generate the first-order logic goal
- [x] Implement the bt tree generator to generate the bt tree
- [x] Implement the Isaaclab simulator, scene usd, robot and its locomotion policy and sensor configuration
- [x] ROS bridge between the Isaacsim5 and ROS2
- [x] Testing dualmap mapping with Isaacsim env 
- [ ] Define and implement the atomic actions and conditions to connect agent and simulator
- [ ] Implement the online open-vacabulary object/place querying
- [ ] Implement the path planner and cmd_vel controller
- [ ] Enhance the prompts engineering with memory / map for better reasoning and interactive task
- [ ] Implement the usage examples for EG_system running in simulator
- [ ] Implement the frontier exploration navigator for task-oriented exploration


## Dependencies
- Python >= 3.11 (only tested on Ubuntu Python 3.11)
- see [requirements.txt](requirements.txt)

> to deploy and test EG_agent in simulation:
- [IsaacSim](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html) == 5.0.0
- [IsaacLab](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html) == 2.2.1


## Usage
module standalone test: [examples](examples/EXAMPLES.md)


## Acknowledgments
- Prompts for well-formed first-order logic goal prompts modified from: https://github.com/HPCL-EI/RoboWaiter.git
- BehaviorTree generator modified from: https://github.com/DIDS-EI/BTPG.git
- Object/Place querying in the unkown world using open-vocabulary mapping: https://github.com/Eku127/DualMap.git
- Simulator: https://github.com/isaac-sim/IsaacLab.git