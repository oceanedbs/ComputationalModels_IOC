import numpy as np
from qpsolvers import solve_qp
import scipy.sparse as sp
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteMDH, PrismaticDH

class Arm4dof:
    def __init__(self, L, M, I, COM, alpha, offset, n):
        self.n = n  # Number of joints
        self.q = np.zeros(self.n)  # Joint angles in radians
        self.qd = np.zeros(self.n) # Joint velocities in rad/s
        self.q_null = np.zeros(self.n) # Desired Nullspace position
        self.W_null = np.eye(self.n) # Weight matrix for nullspace
        self.L = L
        self.M = M
        self.I = I
        self.COM = COM
        self.alpha = alpha
        self.offset = offset    
    
        

    def create_DH_model(self):
        
        list_links = []
        L1 = RevoluteMDH(a=0, m=0, alpha=0, r=[0, 0, 0], modified=True, offset=0)
        list_links.append(L1)
        print(self.n)
        for i in np.arange(0,self.n):
            print(i)
            L = RevoluteMDH(a=self.L[i], m=self.M[i], alpha=self.alpha[i], r=self.COM[:,i], inertia=self.I[i], modified=True, offset=self.offset[i])
            list_links.append(L)
            
        return DHRobot(list_links, name="robot_arm")
        # L2 = RevoluteMDH(a=self.L[0], m=self.M[0], alpha=np.pi / 2, r=self.COM[0], inertia = self.I[0],  modified=True, offset=-np.pi/2)
        # # L3 = RevoluteMDH(a=self.L[1], m=self.M[1], alpha=-np.pi / 2, r=self.COM[1], inertia = self.I[1], modified=True, offset=-np.pi/2)
        # # L4 = RevoluteMDH(a=self.L[2], m=self.M[2], alpha=0, r=self.COM[2], inertia=self.I[2], modified=True)
        # # # L5 = RevoluteMDH(a=self.L[3], d=0, m=self.M[3], alpha=0, r=self.COM[3], inertia=self.I[3], modified=True)

        # # Build the robot
        # self.model = DHRobot([L1, L2]) #, L3, L4], name="ABLE")
        
        # return self.model
        # # self.model_elbow = DHRobot([L1, L2, L3, L4], name="ABLE_elbow")


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
    
    def get_dh_params(self):
        n = self.n
        dh = np.zeros((n, 3))
        L_arr = np.asarray(self.L)
        alpha_arr = np.asarray(self.alpha)

        if L_arr.size == n and alpha_arr.size == n:
            dh[:, 0] = L_arr
            dh[:, 1] = alpha_arr
        elif L_arr.size == n - 1 and alpha_arr.size == n - 1:
            # first row stays zeros (base link), fill remaining rows
            dh[1:, 0] = L_arr
            dh[1:, 1] = alpha_arr
        else:
            raise ValueError("Lengths of self.L and self.alpha must be either n or n-1")

        # last column remains zeros
        return dh
    
    
    
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