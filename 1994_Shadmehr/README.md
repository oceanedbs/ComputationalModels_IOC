# 🧠 Adaptive Representation of Dynamics — Shadmehr & Mussa-Ivaldi (1994)

This repository implements a simulation of **motor adaptation in a two-link planar arm**, inspired by the seminal paper:

> **Reza Shadmehr & Ferdinand A. Mussa-Ivaldi (1994)**  
> *“Adaptive Representation of Dynamics During Learning of a Motor Task”*  
> *Journal of Neuroscience, 14(5), 3208–3224.*

---

## 📖 Overview

This project simulates a **2D planar arm (shoulder + elbow)** controlled by a torque-based feedback system.  
The simulation reproduces **trajectory learning in force fields**, as observed in human reaching experiments.

Key features:
- Minimum-jerk trajectory generation
- Inverse and direct kinematics of a two-link arm
- Torque-based control with stiffness/viscosity feedback
- Simulation of environmental force fields
- Visualization of adaptation and after-effects

---

## 🧩 Model Description

The planar arm is modeled as a **two-link manipulator** defined by:

- Link 1 (upper arm): \( L_1, m_1, I_1, r_1 \)
- Link 2 (forearm): \( L_2, m_2, I_2, r_2 \)

The control law follows **Shadmehr & Mussa-Ivaldi’s motor adaptation model**:

\[
\tau = I(q) \, \ddot{q}_{des} + G(q, \dot{q}) + \hat{E}(q, \dot{q}) - K (q - q_{des}) - V (\dot{q} - \dot{q}_{des})
\]

where:
- \( I(q) \): inertia matrix  
- \( G(q,\dot{q}) \): Coriolis and centrifugal forces  
- \( K, V \): stiffness and viscosity feedback matrices  
- \( \hat{E} \): learned or estimated environmental forces  
- \( q_{des}, \dot{q}_{des}, \ddot{q}_{des} \): desired joint states from inverse kinematics

The arm moves along a **minimum-jerk trajectory** in Cartesian space, then maps it to joint space.

---

## ⚙️ Features Implemented

- ✅ Two-link planar arm dynamics (`I(q)`, `G(q, dq)`)
- ✅ Minimum-jerk trajectory generation (`fun_minjerktrajectory`)
- ✅ Inverse & direct kinematics, Jacobian, and time derivative (`jacobian`, `jacobian_dot`)
- ✅ Torque control law with stiffness/viscosity feedback
- ✅ External and learned force fields (`endpoint_force_field`, `joint_torque_field`)
- ✅ Visualization of trajectories and arm configuration (`plot_arm`, `plot_trajectory`)

---

## 📂 File Structure

.
├── Shadmher1994.py # Main simulation script (controller, dynamics, plots)
├── TwoDLArm_func.py # Helper functions (kinematics, Jacobians, force fields)
├── Shadmehr1994.md # Reference summary and theoretical notes
└── README.md # This documentation


---

## 🧮 Key Functions

| Function | File | Description |
|-----------|------|-------------|
| `inertia_matrix(q, L1, L2, ...)` | TwoDLArm_func.py | Computes joint-space inertia matrix |
| `coriolis_matrix(q, dq, ...)` | TwoDLArm_func.py | Computes Coriolis and centrifugal forces |
| `fun_minjerktrajectory(...)` | TwoDLArm_func.py | Generates minimum-jerk position, velocity, acceleration |
| `jacobian(q, L1, L2)` | TwoDLArm_func.py | Maps joint to Cartesian velocities |
| `jacobian_dot(q, dq, L1, L2)` | TwoDLArm_func.py | Computes time derivative of Jacobian |
| `controller(...)` | Shadmher1994.py | Implements torque-based feedback control law |
| `simulate_movement(...)` | Shadmher1994.py | Runs the full dynamic simulation |
| `endpoint_force_field(x_dot)` | TwoDLArm_func.py | Cartesian force field disturbance |
| `joint_torque_field(q, dq)` | TwoDLArm_func.py | Learned internal torque field |
| `plot_arm(...)` | TwoDLArm_func.py | Draws the two-link arm in workspace |
| `plot_trajectory(...)` | Shadmher1994.py | Plots end-effector movement |

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install numpy scipy matplotlib pandas
```

### 2. Run the main script
```bash
python Shadmher1994.py
```

This will:
- Plot the arm configuration and desired trajectory.
- Simulate free movement (no force field).
- Simulate movement in a velocity-dependent force field.
- Simulate after-effects after learning the field.
- Plot resulting hand trajectories and joint responses.

## 🎨 Visualization

During execution, several plots will appear:

- Desired vs. simulated **hand trajectory**
- Minimum-jerk **velocity and acceleration profiles**
- Desired **joint angles and velocities**
- End-effector **path before, during, and after adaptation**

---

## 🧠 Learning Interpretation

- **No field:** The hand follows the minimum-jerk path.  
- **Force field:** Trajectory deviates due to external forces.  
- **After-effect:** When the field is removed but the learned torque field remains, the trajectory deviates in the opposite direction — a hallmark of motor adaptation.

---

## 📚 Reference

- Shadmehr, R., & Mussa-Ivaldi, F. A. (1994).  
  *Adaptive representation of dynamics during learning of a motor task.*  
  *The Journal of Neuroscience, 14(5), 3208–3224.*  
  [DOI: 10.1523/JNEUROSCI.14-05-03208.1994](https://doi.org/10.1523/JNEUROSCI.14-05-03208.1994)

---

## 🧑‍💻 Author Notes

This code aims to **educate and reproduce** the fundamental concepts of motor adaptation.  
It demonstrates:
- Dynamic simulation of a two-link limb,
- Minimum-jerk motion planning,
- Torque control with stiffness & viscosity,
- Adaptive force-field compensation.

Use it for:
- Teaching human motor control,
- Robotics and biomechanics simulation,
- Comparing adaptive learning models.


---
