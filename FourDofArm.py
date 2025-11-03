import numpy as np
from qpsolvers import solve_qp
import scipy.sparse as sp
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteMDH, PrismaticDH

class Arm:
    def __init__(self, L1, L2):
        self.n = 4  # Number of joints
        self.L1 = L1  # Upper arm length in meters
        self.L2 = L2 # Forearm length in meters
        self.q = np.zeros(self.n)  # Joint angles in radians
        self.qd = np.zeros(self.n) # Joint velocities in rad/s
        self.q_null = np.zeros(self.n) # Desired Nullspace position
        self.W_null = np.eye(self.n) # Weight matrix for nullspace
        

    def create_DH_model(self, arm_mass, forearm_mass, arm_com, forearm_com):
        
        L1 = RevoluteMDH(a=0, m=0, alpha=0, r=[0, 0, 0], modified=True, offset=0)
        L2 = RevoluteMDH(a=0, m=0, alpha=np.pi / 2, r=[0, 0, 0], modified=True, offset=-np.pi/2)
        L3 = RevoluteMDH(a=0, m=0, alpha=-np.pi / 2, r=[0, 0, 0], modified=True, offset=-np.pi/2)
        L4 = RevoluteMDH(a=self.L1, m=arm_mass, alpha=0, r=[arm_com, 0, 0], inertia=[0, (1/12) * arm_mass * self.L1**2, (1/12) * arm_mass * self.L1**2, 0, 0, 0], modified=True)
        L5 = RevoluteMDH(a=self.L2, d=0, m=forearm_mass, alpha=0, r=[forearm_com, 0, 0], inertia=[0, (1/12) * forearm_mass * self.L2**2, (1/12) * forearm_mass * self.L2**2, 0, 0, 0], modified=True)

        # Build the robot
        self.model = DHRobot([L1, L2, L3, L4, L5], name="ABLE")
        self.model_elbow = DHRobot([L1, L2, L3, L4], name="ABLE_elbow")

        print(self.model)
        return self.model


    def jacobian(q, l1, l2):
        """
        ABLE Jacobian
        Computes the Jacobian of ABLE, given a specific joint configuration.
        
        Parameters:
            q : array-like (length 4) -> [q1, q2, q3, q4]
            l1 : float -> shoulder/elbow distance (mm)
            l2 : float -> elbow/end-effector distance (mm)
        
        Returns:
            J : numpy.ndarray (3x4) Jacobian matrix
        """
        q1, q2, q3, q4 = q
        
        J = np.zeros((3, 4))

        # First row
        J[0, 0:4] = [
            - (l1*np.cos(q1)*np.sin(q3))/1000 
            - (l2*np.cos(q4)*(np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)))
            - (l2*np.sin(q4)*(np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3)))
            - (l1*np.cos(q2)*np.cos(q3)*np.sin(q1))/1000,
            
            np.cos(q1)*((l1*np.cos(q3)*np.sin(q2))/1000 
            - (l2*np.sin(q2)*np.sin(q3)*np.sin(q4)) 
            + (l2*np.cos(q3)*np.cos(q4)*np.sin(q2))),
            
            - np.cos(q2)*((l1*np.cos(q1)*np.sin(q3))/1000 
            + (l2*np.cos(q4)*(np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)))
            + (l2*np.sin(q4)*(np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3)))
            + (l1*np.cos(q2)*np.cos(q3)*np.sin(q1))/1000)
            - np.sin(q1)*np.sin(q2)*((l1*np.cos(q3)*np.sin(q2))/1000 
            - (l2*np.sin(q2)*np.sin(q3)*np.sin(q4)) 
            + (l2*np.cos(q3)*np.cos(q4)*np.sin(q2))),
            
            np.sin(q1)*np.sin(q2)*((l2*np.sin(q2)*np.sin(q3)*np.sin(q4)) 
            - (l2*np.cos(q3)*np.cos(q4)*np.sin(q2)))
            - np.cos(q2)*((l2*np.cos(q4)*(np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)))
            + (l2*np.sin(q4)*(np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3))))
        ]
        
        # Second row
        J[1, 0:4] = [
            0,
            np.sin(q1)*((l1*np.cos(q1)*np.sin(q3))/1000 
            + (l2*np.cos(q4)*(np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)))
            + (l2*np.sin(q4)*(np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3)))
            + (l1*np.cos(q2)*np.cos(q3)*np.sin(q1))/1000)
            - np.cos(q1)*((l1*np.sin(q1)*np.sin(q3))/1000 
            + (l2*np.cos(q4)*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)))
            + (l2*np.sin(q4)*(np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3)))
            - (l1*np.cos(q1)*np.cos(q2)*np.cos(q3))/1000),
            
            np.cos(q1)*np.sin(q2)*((l1*np.cos(q1)*np.sin(q3))/1000 
            + (l2*np.cos(q4)*(np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)))
            + (l2*np.sin(q4)*(np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3)))
            + (l1*np.cos(q2)*np.cos(q3)*np.sin(q1))/1000)
            + np.sin(q1)*np.sin(q2)*((l1*np.sin(q1)*np.sin(q3))/1000 
            + (l2*np.cos(q4)*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)))
            + (l2*np.sin(q4)*(np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3)))
            - (l1*np.cos(q1)*np.cos(q2)*np.cos(q3))/1000),
            
            np.cos(q1)*np.sin(q2)*((l2*np.cos(q4)*(np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)))
            + (l2*np.sin(q4)*(np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3))))
            + np.sin(q1)*np.sin(q2)*((l2*np.cos(q4)*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)))
            + (l2*np.sin(q4)*(np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3))))
        ]
        
        # Third row
        J[2, 0:4] = [
            (l1*np.cos(q1)*np.cos(q2)*np.cos(q3))/1000 
            - (l2*np.cos(q4)*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)))
            - (l2*np.sin(q4)*(np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3)))
            - (l1*np.sin(q1)*np.sin(q3))/1000,
            
            np.sin(q1)*((l1*np.cos(q3)*np.sin(q2))/1000 
            - (l2*np.sin(q2)*np.sin(q3)*np.sin(q4)) 
            + (l2*np.cos(q3)*np.cos(q4)*np.sin(q2))),
            
            np.cos(q1)*np.sin(q2)*((l1*np.cos(q3)*np.sin(q2))/1000 
            - (l2*np.sin(q2)*np.sin(q3)*np.sin(q4)) 
            + (l2*np.cos(q3)*np.cos(q4)*np.sin(q2)))
            - np.cos(q2)*((l1*np.sin(q1)*np.sin(q3))/1000 
            + (l2*np.cos(q4)*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)))
            + (l2*np.sin(q4)*(np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3)))
            - (l1*np.cos(q1)*np.cos(q2)*np.cos(q3))/1000),
            
            - np.cos(q2)*((l2*np.cos(q4)*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)))
            + (l2*np.sin(q4)*(np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3))))
            - np.cos(q1)*np.sin(q2)*((l2*np.sin(q2)*np.sin(q3)*np.sin(q4)) 
            - (l2*np.cos(q3)*np.cos(q4)*np.sin(q2)))
        ]
        
        return J
    
    
    
    def compute_mass_matrix(self, model, q, eps=1e-8):
        """Try model.inertia(q) first; if not available, compute M by RNE trick."""
        n = len(q)
        try:
            M = model.inertia(q)   # robotics-toolbox style inertia matrix
            return np.asarray(M)
        except Exception:
            M = np.zeros((n,n))
            zero_qd = np.zeros(n)
            zero_grav = [0,0,0]
            for i in range(n):
                qdd = np.zeros(n); qdd[i] = 1.0
                tau = np.asarray(model.rne(q, zero_qd, qdd, gravity=zero_grav))
                M[:, i] = tau
            return M

    def f_state(self,x, u, model, gravity):
        """Compute f(x,u) = [ qdot; M^{-1}(u - h(q,qdot)) ]"""
        n = len(x)//2
        q = x[:n]
        qd = x[n:]
        # h = C(q,qd)*qd + G(q) -> RNE with zero qdd
        h = np.asarray(model.rne(q, qd, np.zeros(n), gravity=gravity))
        M = self.compute_mass_matrix(model, q)
        # ensure invertible
        Minv = np.linalg.inv(M)
        qdd = Minv.dot(u - h)
        return np.concatenate([qd, qdd])

    def linearize_finite_diff(self, model, q0, qd0, u0, gravity=[0,0,9.81], eps=1e-6):
        n = len(q0)
        x0 = np.hstack([q0, qd0])
        m = 2*n
        # compute nominal f
        f0 = self.f_state(x0, u0, model, gravity)
        # allocate
        A = np.zeros((m, m))
        B = np.zeros((m, n))
        # central differences for A (w.r.t. each state element)
        for i in range(m):
            dx = np.zeros(m)
            dx[i] = eps
            f_plus = self.f_state(x0 + dx, u0, model, gravity)
            f_minus = self.f_state(x0 - dx, u0, model, gravity)
            A[:, i] = (f_plus - f_minus) / (2*eps)
        # central differences for B (w.r.t. each input torque)
        for j in range(n):
            du = np.zeros(n); du[j] = eps
            f_plus = self.f_state(x0, u0 + du, model, gravity)
            f_minus = self.f_state(x0, u0 - du, model, gravity)
            B[:, j] = (f_plus - f_minus) / (2*eps)
        return A, B