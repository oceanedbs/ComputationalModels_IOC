import numpy as np
import math
from scipy.optimize import root
def direct_kinematics(q, L1, L2):
    """
    Compute the (x,y) position of the end-effector given joint angles q.
    q: array-like, [humeraL1ngle_deg, forearm_angle_deg]
    L1: length of upper arm
    L2: length of forearm
    Returns: (x, y) position of the end-effector
    """
    theta1 = np.radians(q[0]) #yaw angle of the arm (eta) 
    theta2 = np.radians(q[1]) # theta arm's elevation
    theta3 = np.radians(q[2]) # #theta3 humeral rotation
    theta4 = np.radians(q[3]) # theta4 wrist extention
    
    # Calculate elbow position
    x_e = -L1 * np.sin(theta1) * np.sin(theta2)
    y_e = L1 * np.cos(theta1) * np.sin(theta2)
    z_e = -L1 * np.cos(theta2)

    # Calculate wrist position components
    term1_x = L2 * np.sin(theta4) * (np.cos(theta3) * np.sin(theta1) * np.cos(theta2) + np.sin(theta3) * np.cos(theta1))
    term2_x = L2 * np.cos(theta4) * np.sin(theta1) * np.sin(theta2)

    term1_y = L2 * np.sin(theta4) * (np.cos(theta3) * np.cos(theta1) * np.cos(theta2) - np.sin(theta3) * np.sin(theta1))
    term2_y = L2 * np.cos(theta4) * np.cos(theta1) * np.sin(theta2)

    term1_z = L2 * np.sin(theta4) * np.cos(theta3) * np.sin(theta2)
    term2_z = L2 * np.cos(theta4) * np.cos(theta2)

    # Calculate wrist position
    x_w = x_e - term1_x + term2_x
    y_w = y_e + term1_y + term2_y
    z_w = z_e + term1_z - term2_z

    return (x_e, y_e, z_e), (x_w, y_w, z_w)
    
    
def inverse_kinematics(target, L1, L2, current_q=None):
    """
    Compute joint angles to reach target (x,y,z) position.
    Uses numerical inverse kinematics (Jacobian transpose method).

    Parameters:
    target: (x,y,z) desired end-effector position
    L1: length of upper arm
    L2: length of forearm
    current_q: current joint angles [eta, theta, phi, zeta] in degrees (optional)

    Returns: joint angles [eta, theta, phi, zeta] in degrees
    """
    # Initial guess if no current angles provided
    if current_q is None:
        q = np.array([45.0, 45.0, 0.0, 0.0])  # Default guess in degrees
    else:
        q = np.array(current_q)

    # Convert to radians for calculations
    q_rad = np.deg2rad(q)

    # Learning rate and tolerance
    alpha = 0.01
    tolerance = 1e-4
    max_iterations = 100000

    for i in range(max_iterations):
        # Get current end-effector position
        (_, _, _), (x_w, y_w, z_w) = direct_kinematics(q, L1, L2)
        current_pos = np.array([x_w, y_w, z_w])

        # Calculate error
        error = target - current_pos
        if np.linalg.norm(error) < tolerance:
            break

        # Compute Jacobian matrix numerically
        J = compute_jacobian(q_rad, L1, L2)

        # Update joint angles using Jacobian transpose
        delta_q = alpha * damped_pseudoinverse(J) @ error
        q_rad += delta_q
        q = np.degrees(q_rad)    
    return q

def compute_jacobian(q, L1, L2):
    """Compute the Jacobian matrix for our arm model"""
    eta, theta,zeta, phi  = q
    q_rad = np.radians(q)

    # Precompute trigonometric values
    s1, c1 = np.sin(eta), np.cos(eta)
    s2, c2 = np.sin(theta), np.cos(theta)
    s3, c3 = np.sin(zeta), np.cos(zeta)
    s4, c4 = np.sin(phi), np.cos(phi)

    
    J = np.zeros((3, 4))
    
    # Partial derivatives w.r.t q1
    J[0, 0] = -L1*c1*s2 + L2*((s1*s3 - c1*c2*c3)*s4 - c1*s2*c4)
    J[1, 0] = -L1*s1*s2 - L2*((s1*c2*c3 + c1*s3)*s4 + s1*s2*c4)
    J[2, 0] = 0.0

    # Partial derivatives w.r.t q2
    J[0, 1] = (-L1*c2 + L2*(s2*s4*c3 - c2*c4))*s1
    J[1, 1] = ( L1*c2 - L2*(s2*s4*c3 - c2*c4))*c1
    J[2, 1] =  L1*s2 + L2*(s2*c4 + s4*c2*c3)

    # Partial derivatives w.r.t q3
    J[0, 2] =  L2*(s1*s3*c2 - c1*c3)*s4
    J[1, 2] = -L2*(s1*c3 + c1*s3*c2)*s4
    J[2, 2] = -L2*s2*s3*s4

    # Partial derivatives w.r.t q4
    J[0, 3] = -L2*((s1*c2*c3 + c1*s3)*c4 - s1*s2*s4)
    J[1, 3] = -L2*((s1*s3 - c1*c2*c3)*c4 + s2*s4*c1)
    J[2, 3] =  L2*(s2*c3*c4 + s4*c2)
    return J


def damped_pseudoinverse(J, damping=1e-8):
    """
    Compute the damped least-squares pseudoinverse of J.
    J: (m x n) numpy array. Returns n x m matrix.
    damping: scalar lambda^2 regularization (use small positive).
    """
    m, n = J.shape
    if m <= n:
        # J^+ = J^T (J J^T + lambda^2 I)^-1
        JJt = J @ J.T
        reg = damping * np.eye(m)
        inv = np.linalg.inv(JJt + reg)
        return J.T @ inv
    else:
        # fallback: (J^T J + lambda^2 I)^-1 J^T
        JtJ = J.T @ J
        reg = damping * np.eye(n)
        inv = np.linalg.inv(JtJ + reg)
        return inv @ J.T

def ik_with_nullspace(target_xyz,
                      q_init_deg,
                      L1, L2,
                      damping=1e-3,
                      fixed_joint_idx=None,
                      fixed_joint_value_deg=None):
    """
    Compute joint angles to reach target (x,y,z) position.
    Uses numerical inverse kinematics (Jacobian transpose method).

    Parameters:
    target: (x,y,z) desired end-effector position
    L1: length of upper arm
    L2: length of forearm
    current_q: current joint angles [eta, theta, phi, zeta] in degrees (optional)

    Returns: joint angles [eta, theta, phi, zeta] in degrees
    """
    # Initial guess if no current angles provided
    if q_init_deg is None:
        q = np.array([45.0, 45.0, 0.0, 0.0])  # Default guess in degrees
    else:
        q = np.array(q_init_deg)
        q_d = np.deg2rad(q.copy())
        if fixed_joint_idx is not None and fixed_joint_value_deg is not None:
            q_d[fixed_joint_idx] = np.deg2rad(fixed_joint_value_deg)

    # Convert to radians for calculations
    q_rad = np.deg2rad(q)

    # Learning rate and tolerance
    alpha = 0.01
    tolerance_dq = 1e-5
    tolerance_f = 1e-4
    max_iterations = 100000

    for i in range(max_iterations):
        # Get current end-effector position
        (_, _, _), (x_w, y_w, z_w) = direct_kinematics(q, L1, L2)
        current_pos = np.array([x_w, y_w, z_w])

        # Calculate error
        error = target_xyz - current_pos
       
        # Compute Jacobian matrix numerically
        J = compute_jacobian(q_rad, L1, L2)
        w = np.diag([0,0,0,0])
        w[fixed_joint_idx, fixed_joint_idx] = 1

        N= np.eye(len(q))- damped_pseudoinverse(J, damping) @ J
        # Update joint angles using Jacobian transpose
        delta_q = alpha * damped_pseudoinverse(J) @ error +500* alpha * N @ w @ (q_d - q_rad)
        q_rad += delta_q
        q = np.degrees(q_rad)
        #print(i, q_d, q_rad, np.linalg.norm(delta_q), np.linalg.norm(error))
        if np.linalg.norm(delta_q) < tolerance_dq or np.linalg.norm(w @ (q_d - q_rad)) < tolerance_f :
            break
    

    return q
