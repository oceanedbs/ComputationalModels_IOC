# 🧠 Computational Models of Inverse Optimal Control (IOC)

This repository contains the implementations, data, and supporting tools for several **Inverse Optimal Control (IOC)** models spanning three decades of research.  
Each folder reproduces or extends a specific published study in the field of motor control and robotics, allowing comparative analysis and experimentation on human and robot motion modeling.

---

## 📁 Repository Structure

.
├── 1994_Shadmehr/       # Code to reproduce the Shadmehr & Mussa-Ivaldi (1994) experiment
├── 2002_Vetter/          # Implementation of the Vetter et al. (2002) model
├── 2022_Colombel/        # IOC model developed by Colombel et al. (2022)
├── 2024_Becanovic/       # IOC implementation by Becanovic et al. (2024, IEEE Humanoids)
├── Data/                 # Experimental and simulation datasets
└── Tools/                # Mathematical reminders and helper scripts


---

## 🧩 Project Overview

This project provides a unified codebase to explore how **Inverse Optimal Control** principles have evolved in computational neuroscience and humanoid robotics.

Each submodule corresponds to a published study, re-implemented for reproducibility and comparison.

| Year | Study / Author | Description | Notes |
|------|----------------|--------------|-------|
| **1994** | Shadmehr & Mussa-Ivaldi | Classic model of human motor adaptation in force fields. | Introduced internal model learning in human arm dynamics. |
| **2002** | Vetter et al. | Refinement of motor control models with predictive coding. | Adds feedforward adaptation mechanisms. |
| **2022** | Colombel et al. | IOC framework applied to human motion capture data. | Focused on estimating cost functions from observed trajectories. |
| **2024** | Becanovic et al. | IOC on humanoid robots at single-level abstraction (IEEE Humanoids 2024). | Extends IOC to robotic control and learning. |

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ComputationalModels_IOC.git
cd ComputationalModels_IOC
```

### 2. Set up the environment
You can install dependencies globally or use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib pandas
``` 

### 3. Run a model
Each experiment can be executed independently:

```bash
cd 1994_Shadmehr
python Shadmehr_1994.py
```
Each subfolder has its own README.md file

## 📚 References

Shadmehr, R., & Mussa-Ivaldi, F. A. (1994). Adaptive representation of dynamics during learning of a motor task. J. Neurosci.

Vetter, P., Flash, T., & Wolpert, D. M. (2002). Planning movements in a simple redundant task. Nature.

Colombel, R., et al. (2022). Inverse Optimal Control of Human Motions.

Becanovic, D., et al. (2024). Single-Level Inverse Optimal Control for Humanoids. IEEE-RAS Humanoids 2024.