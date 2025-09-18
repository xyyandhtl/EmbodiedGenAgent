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
