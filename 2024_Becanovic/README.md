# 🦾 Becanovic 2024 — 2-DOF Robotic System Optimization and Inverse Optimal Control

This repository contains a **Python implementation** of the modeling, trajectory optimization, and **inverse optimal control (IOC)** framework described in *Becanovic et al. (2024)*.  
It uses the [CasADi](https://web.casadi.org/) library for symbolic and numerical optimization, and visualizes results via Matplotlib.
This is a python translation from the original matlab code that can be found [here](https://github.com/Beca-Filip/ioc_planar). 

The corresponding research article is : 
Filip Becanovic, Kosta Jovanović, Vincent Bonnet. Reliability of Single-Level Equality-Constrained Inverse Optimal Control. 2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids), Nov 2024, Nancy, France. pp.623-630, 10.1109/Humanoids58906.2024.10769923. hal-04931691

---

## 📘 Overview

The project simulates an **N-degree-of-freedom (N-DOF)** planar robotic manipulator performing a trajectory optimization task and identifies underlying cost function weights via **inverse optimal control**.

The workflow includes:

1. **Forward Dynamics Modeling** (`Becanovic_2024_func.py`)
2. **Trajectory Optimization** via CasADi’s `Opti` interface (`Becanovic_2024.py`)
3. **Inverse Optimal Control** (estimating cost weights from optimal trajectories)
4. **Visualization** of robot motion, joint trajectories, and segment velocities (`Becanovic_2024_plot.py`)

---

## 🧩 Repository Structure

├── Becanovic_2024.py # Main script: sets up and runs optimization + IOC
├── Becanovic_2024_func.py # Model definitions and helper functions
├── Becanovic_2024_plot.py # Visualization and plotting utilities
└── README.md # This file


---

## ⚙️ Dependencies

You will need the following Python packages:

```bash
pip install numpy matplotlib casadi
```

---

## 🚀 How to run ? 

Run the main simulation and IOC experiment:

```bash
python Becanovic_2024.py
```
This script performs the following steps:

1. Defines the N-DOF model (default: 2 joints).

2. Solves an optimal control problem minimizing a weighted combination of cost terms (e.g., torque, velocity, jerk).

3. Performs IOC to estimate the cost weights (theta) based on the optimal trajectory.

Plots:

- Robot snapshots during motion

- Joint trajectories (q, dq, ddq, and torque)

- Segment velocities

--- 

## 🧠 Key Components

### 1. `make_ndof_model(n, N)`
Defined in `Becanovic_2024_func.py`  
Creates the symbolic CasADi model for an N-DOF robotic arm:
- State variables: `q`, `dq`, `ddq`
- Parameters: link lengths, COM, mass, inertia, gravity, etc.
- Costs: joint torque, velocity, acceleration, jerk, posture, accuracy, etc.
- Constraints: dynamics, initial/goal conditions.

### 2. `instantiate_ndof_model(...)`
Sets the numerical parameter values and initial guesses in the CasADi optimization problem.

### 3. `numerize_var(...)`
Extracts numeric values from symbolic CasADi structures for plotting and analysis.

### 4. `plot_snapshots_from_vars`, `plot_joint_traj_from_vars`, `plot_segment_vels_from_vars`
Defined in `Becanovic_2024_plot.py`  
Functions for visualizing robot configurations and dynamic quantities:
- `plot_snapshots_from_vars`: Draws the robot’s motion snapshots over time.
- `plot_joint_traj_from_vars`: Plots joint position, velocity, acceleration, and torque trajectories.
- `plot_segment_vels_from_vars`: Compares analytical and numerical segment velocities.

### 5. `Becanovic_2024.py`
Main execution script that:
1. Builds the N-DOF model using CasADi.
2. Solves a trajectory optimization problem.
3. Runs the Inverse Optimal Control (IOC) procedure to identify cost weights.
4. Plots robot trajectories and dynamic profiles.

## 📊 Outputs

After running the main script (`Becanovic_2024.py`), the program produces several visual and numerical outputs:

### 🧩 1. Trajectory Plots
Displays the evolution of:
- Joint positions (`q`)
- Joint velocities (`dq`)
- Joint accelerations (`ddq`)
- Joint torques (`τ`)

These are plotted over time using `plot_joint_traj_from_vars()`.

### 🤖 2. Robot Snapshots
The function `plot_snapshots_from_vars()` visualizes snapshots of the robot’s configuration throughout the motion, showing:
- Link positions
- Center of mass (COM) trajectories
- End-effector path

### ⚙️ 3. Segment Velocities
`plot_segment_vels_from_vars()` compares analytically computed and numerically estimated segment velocities along the motion.

### 📈 4. Inverse Optimal Control Results
After optimization, the identified cost weights (`theta_id`) are compared with the true weights (`theta_true`):

```bash
True theta = [0.0002, 1.0000, 0.0002, ...]
Id. theta = [0.0003, 0.9985, 0.0002, ...]
``` 


These results verify that the IOC procedure correctly identifies the underlying cost structure used to generate the motion.
