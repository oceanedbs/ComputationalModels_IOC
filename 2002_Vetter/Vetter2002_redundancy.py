# Also from : Soechting, J F et al. 
# “Moving effortlessly in three dimensions: does Donders' law apply to arm movement?.” 
# The Journal of neuroscience : the official journal of the Society for Neuroscience vol. 15,9 (1995): 6271-80. 
# doi:10.1523/JNEUROSCI.15-09-06271.1995

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from FourDLArm_func import *
import numpy as np
import matplotlib.pyplot as plt
from QP_solver import Arm
from QP_solver import QPController

# ---------------- User parameters ----------------
L1 = 0.30   # upper-arm length (m) - should match your FourDLArm_func params
L2 = 0.30   # forearm length (m)
Kp = 0.5  # Proportional gain for position error
alpha = 0.5 # Scalar for secondary nullspace task         
use_minimal_work = True
start_q = [-20.0, 15.0, 10.0, 40.0]  # starting posture (deg)
                         
arm = Arm(L1, L2)
arm.q = np.array(np.deg2rad(start_q))   # starting posture (deg)
arm.q_null = np.array(np.deg2rad([0.0, 0.0, 0.0, 0.0]))  # desired nullspace posture (deg)
arm.W_null = np.diag([0.0, 1.0, 0.0, 0.0])  # weights for secondary task (higher = more important)

target = np.array([-0.10, 0.3, -0.4])   # target (x,y,z) in meters

# ------------------ Soechting-like parameters (typical values) ------------------
# These are the 'typical' numbers used in Soechting et al. computations (kg, m, kg*m^2)
params = {
    # segment lengths (m)
    'L_upper': L1,   # example upper-arm length
    'L_fore': L2,   # example forearm length
    # masses (kg)
    'm_upper': 1.8,    # approximate
    'm_fore':  1.2,
    # distances from proximal joint to segment COM (m)
    'a_upper': 0.135,
    'a_fore':  0.184,
    # moments (approximate about COM) -- paper gives sample values; you can tune
    'I_upper_perp': 0.312,   # inertia of upper arm about axes perpendicular to long axis
    'I_upper_long': 0.003,   # inertia about long (humeral) axis (much smaller)
    'I_fore_perp': 0.123,
    'I_fore_long': 0.0026,
    # scalar to scale peak angular-velocity proportionality (cancels out if same for all candidates)
    'omega_scale': 1.0
}

# trajectory params
T = 0.6
dt = 0.01


def minimal_work(q, q_dot, I_a1, I_a2, I_f1, I_f2,
                   m_a, m_f, l_a, a_a, a_f):
    """
    Compute the kinetic energy W of the system.

    Parameters:
    eta, theta, zeta: angles in radians
    eta_dot, theta_dot, zeta_dot: angular velocities in rad/s
    I_a1, I_a2: moments of inertia for the arm
    I_f1, I_f2: moments of inertia for the forearm
    m_a, m_f: masses of the arm and forearm
    l_a: length of the arm
    a_a, a_f: distances to the centers of mass of the arm and forearm

    Returns:
    W: kinetic energy
    """
    eta, theta, zeta = np.radians(q)
    eta_dot, theta_dot, zeta_dot = np.radians(q_dot)

    # Compute composite inertia terms
    I1 = I_a1 + m_a * a_a**2 + m_f * l_a**2
    I2 = I_a2
    I3 = I_f1 + m_f * a_f**2
    I4 = I_f2
    A = m_f * l_a * a_f

    # Compute Omega terms
    Omega_x = eta_dot * np.sin(zeta) * np.sin(theta) + theta_dot * np.cos(zeta)
    Omega_y = eta_dot * np.cos(zeta) * np.sin(theta) - theta_dot * np.sin(zeta)
    Omega_z = eta_dot * np.cos(theta) + zeta_dot

    # Compute the kinetic energy W
    term1 = 0.5 * I1 * (eta_dot**2 * np.sin(theta)**2 + theta_dot**2)

    term2 = I2 * (eta_dot * np.cos(theta) + zeta_dot)**2

    term3 = I3 * (Omega_x**2 + Omega_y**2 * np.cos(zeta)**2 +
                  Omega_y**2 * np.sin(zeta)**2 + zeta_dot**2 +
                  2 * zeta_dot * Omega_x)

    term4 = 2 * Omega_z * Omega_y * np.cos(zeta) * np.sin(zeta)

    term5 = I4 * (Omega_y**2 * np.sin(zeta)**2 + Omega_y**2 * np.cos(zeta)**2 -
                  2 * Omega_z * Omega_y * np.cos(zeta) * np.sin(zeta))

    term6 = A * (Omega_y * np.cos(zeta) + Omega_x * np.cos(zeta) + Omega_z * Omega_y * np.sin(zeta))

    W = term1 + term2 + term3 + term4 + term5 + term6

    return W


# Standard inverse kinematics gradient descent
# Compute arm points for start_q
arm_pts = direct_kinematics(start_q, params['L_upper'], params['L_fore'])
hand_pos = arm_pts[1]
elb_pos = arm_pts[0]
shou_pos = (0.0, 0.0, 0.0)

#Compute final arm configuration via IK
ik_sol = inverse_kinematics(target, params['L_upper'], params['L_fore'], current_q=start_q)

# Compute arm points for start_q
final_pos = direct_kinematics(ik_sol, params['L_upper'], params['L_fore'])
hand_final = final_pos[1]
elb_final = final_pos[0]

# Work minimization
fig = plt.figure()

work = []
for i in np.arange(40, 60, 1):
    print(i)
    arm.q_null[1] = np.deg2rad(i)

    # Calculate inverse kinematics with QP solver for gradient descent
    controller = QPController(arm, dt=dt)
    arrived = False
    while not arrived:
        # Compute joint velocities to move towards target
        xdot = Kp * (target - arm.fkine(arm.q)) # desired end-effector velocity is proportionnal to the position error
        controller.update_robot_state(arm) # update robot state for fkine and jacobian calc
        controller.solve(xdot, alpha = alpha) # solve QP to get joint velocities
        qd = controller.solution
        arm.q += qd * dt
        # Check if the end-effector is close enough to the target
        eff_pos = arm.fkine(arm.q)
        if np.linalg.norm(eff_pos - target) < 0.001 and np.linalg.norm(qd) < 1e-3:  # tolerance of 0.1 cm
            arrived = True
            
  
    q1, q2, q3, q4 = np.rad2deg(arm.q)  
    print(q1, q2, q3, q4)
    # Store intermediate solutions for plotting
    interm_solution = direct_kinematics(arm.q, params['L_upper'], params['L_fore'])
    hand_pos_interm = interm_solution[1]
    elb_pos_interm = interm_solution[0]
    
    w = minimal_work((q1, q2, q3), (q1 - start_q[0], q2 - start_q[1], q3 - start_q[2]),
                     params['I_upper_perp'], params['I_upper_long'],
                     params['I_fore_perp'], params['I_fore_long'], params['m_upper'], params['m_fore'],
                     params['L_upper'], params['a_upper'], params['a_fore'])
    work.append((q1, q2, q3, q4, w))
    print(w)
work = np.array(work, dtype=object)
# Find the minimal value in work[:, 1] and get the corresponding configuration from work[:, 0]
min_idx = np.argmin(work[:, 4])
final_pos = [work[min_idx, 0], work[min_idx, 1], work[min_idx, 2], work[min_idx, 3]]
print("Minimal work configuration (eta, theta, zeta, phi), work):", work[min_idx])


# Visualisation inverse kinematics 
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Plot shoulder position
ax.scatter(shou_pos[0], shou_pos[1], shou_pos[2], s=100, marker='o', color='b', label='Shoulder (computed)')

# Plot elbow position
ax.scatter(elb_pos[0], elb_pos[1], elb_pos[2], s=150, marker='o', color='orange', label='Elbow', edgecolors='k')
# Plot a vertical segment along z at the shoulder position
ax.plot([shou_pos[0], shou_pos[0]], [shou_pos[1], shou_pos[1]], [shou_pos[2], shou_pos[2]-0.6], color='purple', linewidth=2, label='Vertical (z) segment')
# Optionally, connect the points to show the arm segments
ax.plot([shou_pos[0], elb_pos[0]], [shou_pos[1], elb_pos[1]], [shou_pos[2], elb_pos[2]], color='gray', linewidth=2)
ax.plot([elb_pos[0], hand_pos[0]], [elb_pos[1], hand_pos[1]], [elb_pos[2], hand_pos[2]], color='gray', linewidth=2)

# Plot hand (start) position as a circle
ax.scatter(hand_pos[0], hand_pos[1], hand_pos[2], s=200, marker='o', color='g', label='Hand (start)', edgecolors='k')

# Plot target as a red circle
ax.scatter(target[0], target[1], target[2], s=200, marker='o', color='r', label='Target', edgecolors='k')

# Plot shoulder
ax.scatter(0,0,0, s=100, marker='o', color='k', label='Shoulder')

# Plot elbow position (final)
ax.scatter(elb_final[0], elb_final[1], elb_final[2], s=150, marker='^', color='cyan', label='Elbow (final)', edgecolors='k')
# Connect shoulder to final elbow
ax.plot([shou_pos[0], elb_final[0]], [shou_pos[1], elb_final[1]], [shou_pos[2], elb_final[2]], color='blue', linewidth=2, linestyle='--')
# Connect final elbow to final hand
ax.plot([elb_final[0], hand_final[0]], [elb_final[1], hand_final[1]], [elb_final[2], hand_final[2]], color='blue', linewidth=2, linestyle='--')
# Plot hand (final) position
ax.scatter(hand_final[0], hand_final[1], hand_final[2], s=200, marker='^', color='magenta', label='Hand (final)', edgecolors='k')

# Set equal aspect ratio for all axes
ax.axis('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Arm Configuration and Target with inverse kinematics')
ax.legend()
plt.show()


# Visualisation minimal work

# Highlight the point of minimal work on the plot
plt.figure()
plt.plot(work[:, 1], work[:, 4])
plt.xlabel('Phi (deg)')
plt.ylabel('Minimal Work')
plt.title('Minimal Work vs. Phi')
plt.grid(True)
# Add a red marker at the minimal work point
plt.scatter(work[min_idx, 1], work[min_idx, 4], color='red', s=100, label='Min Work')
plt.legend()
plt.show()

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Plot initial hand position
ax.scatter(hand_pos[0], hand_pos[1], hand_pos[2], s=200, marker='o', color='g', label='Hand (start)', edgecolors='k')

# Connect shoulder to initial elbow
ax.plot([shou_pos[0], elb_pos[0]], [shou_pos[1], elb_pos[1]], [shou_pos[2], elb_pos[2]], color='gray', linewidth=2, label='Upper Arm (start)')
# Connect initial elbow to initial hand
ax.plot([elb_pos[0], hand_pos[0]], [elb_pos[1], hand_pos[1]], [elb_pos[2], hand_pos[2]], color='gray', linewidth=2, label='Forearm (start)')

# Plot final position found by minimal work (final_pos)
final_hand = direct_kinematics(final_pos, params['L_upper'], params['L_fore'])[1]
ax.scatter(final_hand[0], final_hand[1], final_hand[2], s=200, marker='^', color='magenta', label='Hand (min work)', edgecolors='k')

# Plot target as a red circle
ax.scatter(target[0], target[1], target[2], s=200, marker='o', color='r', label='Target', edgecolors='k')

# Plot shoulder
ax.scatter(0,0,0, s=100, marker='o', color='k', label='Shoulder')
# Plot a vertical segment representing the torso at the shoulder position
ax.plot([shou_pos[0], shou_pos[0]], [shou_pos[1], shou_pos[1]], [shou_pos[2], shou_pos[2]-0.6], color='brown', linewidth=4, label='Torso (vertical)')

# Compute elbow position for minimal work configuration
final_elb = direct_kinematics(final_pos, params['L_upper'], params['L_fore'])[0]

# Connect shoulder to elbow (min work)
ax.plot([0, final_elb[0]], [0, final_elb[1]], [0, final_elb[2]], color='blue', linewidth=2, linestyle='--', label='Upper Arm (min work)')
# Connect elbow to hand (min work)
ax.plot([final_elb[0], final_hand[0]], [final_elb[1], final_hand[1]], [final_elb[2], final_hand[2]], color='blue', linewidth=2, linestyle='--', label='Forearm (min work)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.axis('equal')
ax.set_zlabel('Z (m)')
ax.set_title('Initial and Minimal Work Final Hand Positions with Minimal Work Configuration')
ax.legend()
plt.show()
