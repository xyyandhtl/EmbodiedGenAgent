# Go2W Locomotion Simulation Demo

## 1. Overview

This document describes the simulation demo for the Unitree Go2W wheeled-legged robot, implemented using NVIDIA Isaac Lab.

The demo showcases a hierarchical control system where a pre-trained low-level locomotion policy follows high-level velocity commands to achieve navigation tasks.

## 2. Features

- **High-Fidelity Simulation**: Utilizes NVIDIA Isaac Lab for realistic physics and rendering.
- **Hierarchical Control**: Decouples high-level task planning (e.g., navigation) from low-level motion execution (e.g., leg and wheel coordination).
- **Pre-trained Locomotion Policy**: Employs a robust, pre-trained policy to handle the complex dynamics of wheeled-legged locomotion.
- **Dual Control Modes**: Supports two distinct modes for commanding the robot:
    1.  **Interactive Keyboard Control**: Allows for direct, real-time control of the robot's velocity.
    2.  **Autonomous Goal-Point Navigation**: A simple high-level controller guides the robot to a predefined target position.

## 3. System Architecture

The control system is divided into two main layers:

#### a. High-Level Controller

This layer is responsible for generating 2D velocity commands (linear velocity `vx` and angular velocity `wz`). The behavior of this layer is determined by the `controller` setting in `settings.yaml`.

- **Keyboard Mode (`controller: keyboard`)**: It uses the `Se2Keyboard` class from Isaac Lab to map keyboard inputs (`up, down, left, right, z, x`) to velocity commands.
- **Goal-Point Mode (`controller: goal_point`)**: A simple Proportional-controller implemented in `mdp.compute_velocity_with_goalPoint` calculates the required velocity to face and move towards a hardcoded `goal_position` defined in `run_demo.py`.

#### b. Low-Level Controller

This layer consists of a pre-trained locomotion policy loaded from a `.jit` file (`policy_roughRecover.jit`).

- **Input**: It takes a series of observations, including the robot's sensor readings (joint positions, velocities, base angular velocity) and the high-level velocity command.
- **Logic**: In each simulation step, the high-level velocity command is explicitly written into the observation vector that is fed to the policy.
- **Output**: The policy outputs target joint positions, which are then sent to the robot's actuators.

## 4. File Structure

Key files and directories within the `simulation/` folder:

- `run_demo.py`: The main entry point for launching the simulation.
- `settings.yaml`: The primary configuration file, managed by Hydra. It's used to set the control mode, simulation UI, and other parameters.
- `env/go2w_locomotion_env_cfg.py`: The Isaac Lab environment configuration. It defines the robot asset, sensors, observation space, action space, and other simulation-related settings.
- `mdp/`: Contains logic related to the Markov Decision Process (MDP), including:
    - `commands.py`: Implements the high-level goal-point navigation logic.
    - `observation.py`: Defines custom observation functions.
- `assets/`: Contains the robot (Unitree Go2W) and terrain assets for the simulation.
- `data/ckpts/`: This directory is intended to store the pre-trained policy models (`.jit` files).

## 5. How to Run

1.  **Navigate** to the project root directory.
2.  **Configure the Mode**: Open `simulation/settings.yaml` and set the desired `controller` mode.
    - For interactive control: `controller: "keyboard"`
    - For autonomous navigation: `controller: "goal_point"`
3.  **Run the script**:
    ```bash
    python simulation/run_demo.py
    ```

### Control Keys (Keyboard Mode)

- **L**: Reset all commands
- **Arrow Up / Numpad 8**: Move forward
- **Arrow Down / Numpad 2**: Move backward
- **Arrow Right / Numpad 4**: Strafe left
- **Arrow Left / Numpad 6**: Strafe left
- **Z / Numpad 7**: Turn left
- **X / Numpad 9**: Turn right


## 6. Configuration

The primary configuration is done in `simulation/settings.yaml`:

- `controller`: Sets the high-level control mode.
  - `"keyboard"`: For manual control.
  - `"goal_point"`: For autonomous navigation to a fixed point.
- `sim_app.headless`:
  - `False`: Run with the full graphical user interface.
  - `True`: Run in headless mode (no UI). This is useful for training or running on a server.
- `policy_device`: The compute device (`cpu` or `cuda:0`) on which the policy will be executed.


## 7. ROS2 Data Communication

To integrate with external modules like `vlmap`, the simulation provides a real-time stream of sensor data (RGB, Depth, Pose) via ROS2 topics.

### a. Architecture: The ZMQ Bridge

A direct integration is not possible due to a Python version conflict: Isaac Sim 5.0 requires **Python 3.11**, while ROS2 Humble is built for **Python 3.10**. 

To solve this, we use a **ZMQ Bridge** architecture, which decouples the simulation environment from the ROS environment.

1.  **`run_demo.py` (Python 3.11)**: The main simulation script. It is **not** a ROS node. It uses the `ZMQDataPublisher` utility to send serialized sensor data over a ZMQ network socket.

2.  **`ros_bridge.py` (Python 3.10)**: A standalone script that runs in a separate, ROS-enabled environment. It acts as a ZMQ subscriber to receive data from the simulation and as a ROS2 publisher to broadcast that data to the ROS network.

3.  **`vlmap` module (Python 3.10)**: The final consumer, which subscribes to the ROS2 topics published by the `ros_bridge.py` script.

This design allows each component to run in its required Python environment while communicating efficiently.

### b. How to Run and Test

Follow these steps to launch the full system and verify the data communication.

#### Prerequisites

1.  **Two Conda Environments**: 
    - One for Isaac Sim with **Python 3.11**.
    - One for VLMAP and ROS with **Python 3.10**.
2.  **Install `pyzmq`**: Ensure `pyzmq` is installed in **both** conda environments:
    ```bash
    # In the Python 3.11 env
    conda activate <your-isaac-env-name>
    pip install pyzmq

    # In the Python 3.10 env
    conda activate <your-ros-env-name>
    pip install pyzmq
    ```

#### Execution Steps

You will need to open **at least three terminals**.

**Terminal 1: Start the `vlmap` Subscriber**

```bash
# Activate ROS environment
conda activate <your-ros-py3.10-env>

# Run the vlmap node
python EG_agent/vlmap/ros_runner/runner_ros.py
```

**Terminal 2: Start the ROS Bridge**

```bash
# Activate ROS environment
conda activate <your-ros-py3.10-env>

# Run the bridge script
python simulation/ros_bridge.py
```

**Terminal 3: Start the Isaac Sim Simulation**

```bash
# Activate Isaac Sim environment
conda activate <your-isaac-py3.11-env>

# Run the main demo script
python simulation/run_demo.py
```

#### Verification Steps

Open a **fourth terminal** with the ROS environment activated.

1.  **List ROS2 Topics**:
    ```bash
    ros2 topic list
    ```
    *Expected*: You should see `/camera/rgb/image_raw`, `/camera/depth/image_raw`, `/camera/pose`, `/camera_info`, and `/tf` among the listed topics.

2.  **Check TF Transform**:
    ```bash
    ros2 run tf2_ros tf2_echo odom camera_link
    ```
    *Expected*: After a brief "Waiting for transform" message, you should see a continuous stream of translation and rotation data.

3.  **Visualize Camera Feed**:
    ```bash
    rqt
    ```
    *Choose*: Plugins → Visualization → Image View

    *Expected*: A graphical window will open. Select `/camera/rgb/image_raw` or  `/camera/depth/image_raw` from the dropdown to see the robot's live point-of-view.
