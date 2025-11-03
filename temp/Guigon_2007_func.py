import numpy as np
from qpsolvers import solve_qp
import scipy.sparse as sp
import matplotlib.pyplot as plt

class Arm:
    def __init__(self, L1, L2):
        self.n = 4  # Number of joints
        self.L1 = L1  # Upper arm length in meters
        self.L2 = L2 # Forearm length in meters
        self.q = np.zeros(self.n)  # Joint angles in radians
        self.qd = np.zeros(self.n) # Joint velocities in rad/s
        self.q_null = np.zeros(self.n) # Desired Nullspace position
        self.W_null = np.eye(self.n) # Weight matrix for nullspace
        
    def fkine(self, q):
        """
        Compute the (x,y) position of the end-effector given joint angles q in radians.
        """
        # Calculate elbow position
        x_e = -self.L1 * np.sin(q[0]) * np.sin(q[1])
        y_e = self.L1 * np.cos(q[0]) * np.sin(q[1])
        z_e = -self.L1 * np.cos(q[1])

        # Calculate wrist position components
        term1_x = self.L2 * np.sin(q[3]) * (np.cos(q[2]) * np.sin(q[0]) * np.cos(q[1]) + np.sin(q[2]) * np.cos(q[0]))
        term2_x = self.L2 * np.cos(q[3]) * np.sin(q[0]) * np.sin(q[1])
        term1_y = self.L2 * np.sin(q[3]) * (np.cos(q[2]) * np.cos(q[0]) * np.cos(q[1]) - np.sin(q[2]) * np.sin(q[0]))
        term2_y = self.L2 * np.cos(q[3]) * np.cos(q[0]) * np.sin(q[1])
        term1_z = self.L2 * np.sin(q[3]) * np.cos(q[2]) * np.sin(q[1])
        term2_z = self.L2 * np.cos(q[3]) * np.cos(q[1])

        # Calculate wrist position
        x_w = x_e - term1_x + term2_x
        y_w = y_e + term1_y + term2_y
        z_w = z_e + term1_z - term2_z

        return np.array([x_w, y_w, z_w])
    
    def fkine_elbow(self, q):
        """
        Compute the (x,y,z) position of the elbow given joint angles q in radians.
        """
        # Calculate elbow position
        x_e = -self.L1 * np.sin(q[0]) * np.sin(q[1])
        y_e = self.L1 * np.cos(q[0]) * np.sin(q[1])
        z_e = -self.L1 * np.cos(q[1])

        return np.array([x_e, y_e, z_e])
    
    def jacobe(self, q):
        """
        Compute the Jacobian matrix of the end-effector at joint angles q in radians.
        """
        q[0], q[1], q[2], q[3]  = q

        # Precompute trigonometric values
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s2, c2 = np.sin(q[1]), np.cos(q[1])
        s3, c3 = np.sin(q[2]), np.cos(q[2])
        s4, c4 = np.sin(q[3]), np.cos(q[3])

        J = np.zeros((3, 4))
        
        # Partial derivatives w.r.t q1
        J[0, 0] = -self.L1*c1*s2 + self.L2*((s1*s3 - c1*c2*c3)*s4 - c1*s2*c4)
        J[1, 0] = -self.L1*s1*s2 - self.L2*((s1*c2*c3 + c1*s3)*s4 + s1*s2*c4)
        J[2, 0] = 0.0

        # Partial derivatives w.r.t q2
        J[0, 1] = (-self.L1*c2 + self.L2*(s2*s4*c3 - c2*c4))*s1
        J[1, 1] = ( self.L1*c2 - self.L2*(s2*s4*c3 - c2*c4))*c1
        J[2, 1] =  self.L1*s2 + self.L2*(s2*c4 + s4*c2*c3)

        # Partial derivatives w.r.t q3
        J[0, 2] =  self.L2*(s1*s3*c2 - c1*c3)*s4
        J[1, 2] = -self.L2*(s1*c3 + c1*s3*c2)*s4
        J[2, 2] = -self.L2*s2*s3*s4

        # Partial derivatives w.r.t q4
        J[0, 3] = -self.L2*((s1*c2*c3 + c1*s3)*c4 - s1*s2*s4)
        J[1, 3] = -self.L2*((s1*s3 - c1*c2*c3)*c4 + s2*s4*c1)
        J[2, 3] =  self.L2*(s2*c3*c4 + s4*c2)
        return J
    
    def plot(self, ax, color="b"):
        # Visualisation inverse kinematics 
        shou_pos = np.array([0,0,0])
        elb_pos = self.fkine_elbow(self.q)
        hand_pos = self.fkine(self.q)

        # Plot shoulder
        ax.scatter(shou_pos[0], shou_pos[1], shou_pos[2], s=100, marker='o', color=color, label='Shoulder (computed)')

        # Plot elbow
        ax.scatter(elb_pos[0], elb_pos[1], elb_pos[2], s=150, marker='o', color=color, label='Elbow', edgecolors='k')
        ax.plot([shou_pos[0], shou_pos[0]], [shou_pos[1], shou_pos[1]], [shou_pos[2], shou_pos[2]-0.6], color=color, linewidth=2, label='Vertical (z) segment')
        ax.plot([shou_pos[0], elb_pos[0]], [shou_pos[1], elb_pos[1]], [shou_pos[2], elb_pos[2]], color=color, linewidth=2)
        ax.plot([elb_pos[0], hand_pos[0]], [elb_pos[1], hand_pos[1]], [elb_pos[2], hand_pos[2]], color=color, linewidth=2)

        # Plot hand (start)
        ax.scatter(hand_pos[0], hand_pos[1], hand_pos[2], s=200, marker='o', color=color, label='Hand (start)', edgecolors='k')
