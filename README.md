# EmbodiedGenAgent
A Generative Embodied Agent that transforms open-ended instructions into behavior trees for autonomous decision-making and action execution loop

## Plans
- [x] Implement the intent understanding: vlm prompt/reflective feedback to generate the first-order logic goal
- [x] Implement the bt tree generator to generate the bt tree
- [ ] Implement the Isaaclab simulator, scene usd, robot and its locomotion policy
- [ ] Define and implement the atomic actions and conditions to connect agent and simulator
- [ ] Implement online/offline dualmap generation
- [ ] Implement the path planner and cmd_vel controller
- [ ] Enhance the prompts engineering with memory and map for better reasoning
- [ ] Implement the usage examples for EG_system running in simulator
- [ ] Implement the frontier exploration navigator for task-oriented exploration

## Usage
module standalone test: [examples](examples/EXAMPLES.md)


## Acknowledgments
- Prompts for well-formed first-order logic goal prompts modified from: https://github.com/HPCL-EI/RoboWaiter.git
- BehaviorTree generator modified from: https://github.com/DIDS-EI/BTPG.git
- Object/Place querying in the unkown world using open-vocabulary mapping: https://github.com/Eku127/DualMap.git
- Simulator: https://github.com/isaac-sim/IsaacLab.git