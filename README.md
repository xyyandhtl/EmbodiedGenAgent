# EmbodiedGenAgent
A Generative Embodied Agent that transforms open-ended instructions into behavior trees for autonomous decision-making and action execution loop

## Plans
- Implement the intent understanding: vlm prompt/reflective feedback to generate the first-order logic goal
- Implement the bt tree generator to generate the bt tree
- Implement the Isaaclab simulator, scene usd, robot and its locomotion policy
- Define the atomic actions and observations interface to connect agent and simulator
- Implement the frontier exploration navigator and semantic/topo/traversability map generation
- Implement the simple path planner and cmd_vel controller from WALK action
- Implement the prompts engineering with memory and map
- Implement the usage examples for bt tree execution loop in simulator

## Usage
module standalone test: [examples](examples/EXAMPLES.md)


## Acknowledgments
- Prompts for well-formed first-order logic goal prompts modified from: https://github.com/HPCL-EI/RoboWaiter.git
- BehaviorTree generator modified from: https://github.com/DIDS-EI/BTPG.git
- Object/Place querying in the unkown world using open-vocabulary mapping: https://github.com/Eku127/DualMap.git
- Simulator: https://github.com/isaac-sim/IsaacLab.git