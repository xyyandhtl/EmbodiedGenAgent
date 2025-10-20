# EmbodiedGenAgent
A Generative Embodied Agent that transforms open-ended instructions into behavior trees for autonomous decision-making and action execution loop

## Usage
- GUI demo: [APP.md](docs/APP.md), need EGAgentSystem as backend, where should implement its deployment details, check [envs](EG_agent/system/envs)

- Agent introduction: [EG_agent.md](docs/EG_agent.md)

- Agent module unit test: [EXAMPLES.md](docs/EXAMPLES.md)

- Agent system isaacsim deployment: [SIMULATION.md](docs/SIMULATION.md), a detailed example of the env implementation

After checking the above guides, here are the overall steps for isaacsim demo:
| 环境 | 命令 | 说明 | 图示 |
|------|------|------|--------|
| **Python 3.10**<br>(Agent + GUI 环境) | `python app.py` 启动后点击`创建后台` | 启动 Agent 与可视化界面，系统将等待机器人 ROS2 话题：<br>`/camera/rgb/image_raw`, `/camera/depth/image_raw`, `/camera_pose`, `/camera_info`（可选），并创建 ROS2 发布话题：<br>`/cmd_vel`, `/global_plan` 等。 | ![agent_gui](docs/assets/gui.jpg) |
| **Python 3.11**<br>(Isaac Sim 5.0 环境) | `python simulation/run_demo.py` | 启动仿真环境。 | ![isaacsim](docs/assets/simulator.jpg) |
| **Python 3.10**<br>(桥接环境，可与 Agent 环境共用) | `python simulation/ros_bridge.py` | 启动 Isaac Sim 与 ROS2 之间的数据桥接。 | ![ros_bridge](docs/assets/rqt.jpg) |

### Guide
- 创建后台: 点击`创建后台`按钮
- 智能体开始工作：点击`启动智能体`按钮
- 载入地图：点击`载入地图`按钮，选择预建地图目录
- 发送指令：在输入框中输入指令，点击`发送`按钮。任务目标会自动提取，请发送和场景有关的指令，若为场景无关指令，则可能会在对话窗口提示指令失败。解析和规划成功后，会在行为树窗口可视化执行行为树，机器人自动开始执行。


## Plans
- [x] Implement the intent understanding: vlm prompt / reflective feedback to generate the first-order logic goal
- [x] Wrap the bt tree generator to generate executable bt from the first-order logic goal
- [x] Implement the Isaaclab simulator, scene usd, robot and its locomotion policy and sensor configuration
- [x] ROS bridge between the Isaacsim5 and ROS2
- [x] Testing dualmap mapping with Isaacsim env 
- [x] Define and implement the atomic actions and conditions to connect agent and simulator
- [x] Wrap the online open-vacabulary object/place querying
- [x] Implement the path planner and cmd_vel controller
- [ ] Enhance the prompts engineering with memory / map for better reasoning and interactive task
- [ ] Usage examples for EG_system running in simulator
- [x] An interactive GUI to demo
- [ ] The frontier exploration navigator for task-oriented exploration


## Dependencies
EG_agent:
- Python == 3.10 (to use the released rclpy without rebuilding ROS2’s non-py310 bindings)
- see [requirements.txt](requirements.txt)

to deploy and test EG_agent in simulation:
- Python >= 3.11 (use a ZMQ ros bridge to connect with EG_agent)
- [IsaacSim](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html) == 5.0.0
- [IsaacLab](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html) == 2.2.1


## Acknowledgments
- Prompts for well-formed first-order logic goal prompts modified from: https://github.com/HPCL-EI/RoboWaiter.git
- BehaviorTree generator modified from: https://github.com/DIDS-EI/BTPG.git
- Object/Place querying in the unkown world using open-vocabulary mapping: https://github.com/Eku127/DualMap.git
- Simulator: https://github.com/isaac-sim/IsaacLab.git