from tracemalloc import stop
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import solve
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.linalg import solve as linsolve
from scipy.linalg import svd
from numpy.linalg import lstsq
import math
from functools import lru_cache

# Constants 
m1, m2 = 1.0, 1.5          # Masses [kg]
l1, l2 = 1.0, 1.2          # Lengths [m]
r1, r2 = 0.5, 0.6          # COM distances [m]
I1, I2 = 0.5, 0.7          # Inertias [kg·m²]
g = 9.81                   # Gravity [m/s²]

Nfeval = 1

# Intermediate terms for M(θ) and G(θ, dθ)
a1 = m1*r1**2 + m2*(l1**2 + r2**2) + I1 + I2
a2 = m2*l1*r2
a3 = m2*r2**2 + I2

# Intermediate terms for g(θ)
b1 = l1*m2 + r1*m1
b2 = r2*m2

# Goal state: [π/2, π/2, 0, 0, 0, 0]
goal = np.array([1.5, 0.6, 0, 0, 0, 0])
# Initial state: [0, 0, 0, 0, 0, 0]
start = np.array([-math.pi/2 + 0.1, -0.1, 0, 0, 0, 0])

n_steps = 40
dt = 0.01

ROOT = os.path.dirname(__file__)
print(ROOT)

@lru_cache(maxsize=None)
def M_cached(theta1, theta2):
    return np.array([
        [a1 + 2*a2*np.cos(theta2), a3 + a2*np.cos(theta2)],
        [a3 + a2*np.cos(theta2),    a3]
    ])

@lru_cache(maxsize=None)
def R_cached(theta1, theta2, dtheta1, dtheta2):
    return np.array([
        [-a2*dtheta2*np.sin(theta2), -a2*(dtheta1 + dtheta2)*np.sin(theta2)],
        [ a2*dtheta1*np.sin(theta2),  0]
    ])

@lru_cache(maxsize=None)
def g_vec_cached(theta1, theta2):
    return np.array([
        b1*g*np.cos(theta1) + b2*g*np.cos(theta1 + theta2),
        b2*g*np.cos(theta1 + theta2)
    ])
    
    
# --- helper analytic derivatives for M, R, g (use your constants a1,a2,a3,b1,b2,g) ---
@lru_cache(maxsize=None)
def dM_dtheta2(theta2):
    # returns 2x2 matrix dM/dθ2
    s = -a2 * np.sin(theta2)
    return s * np.array([[2.0, 1.0],
                         [1.0, 0.0]])
    
@lru_cache(maxsize=None)
def dR_dtheta2(theta2, d1, d2):
    # returns 2x2 matrix ∂R/∂θ2
    c = -a2 * np.cos(theta2)
    # careful with signs: derived in explanation
    return c * np.array([[ d2, (d1 + d2)],
                         [-d1, 0.0]])
    
@lru_cache(maxsize=None)
def dR_dd1(theta2):
    s = a2 * np.sin(theta2)
    return np.array([[0.0, -s],
                     [ s,  0.0]])

@lru_cache(maxsize=None)
def dR_dd2(theta2):
    s = -a2 * np.sin(theta2)
    return np.array([[s, s],
                     [0.0, 0.0]])

@lru_cache(maxsize=None)
def dg_dtheta(theta1, theta2):
    # returns two column vectors: dg/dtheta1 (2,) and dg/dtheta2 (2,)
    s12 = np.sin(theta1 + theta2)
    return (np.array([-b1*g*np.sin(theta1) - b2*g*s12,
                      -b2*g*s12]),
            np.array([-b2*g*s12,
                      -b2*g*s12]))

def direct_kinematics(q, L1, L2):
    """Calcule la position (x, y) du point final à partir de q."""
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])

def inverse_kinematics(x, y, L1, L2):
    """Calcule q1 et q2 pour atteindre (x, y)."""
    # Distance à la cible
    d2 = (x**2 + y**2)
    u = (d2 - L1**2 - L2**2)/(2*L1*L2)
    u = max(min(u, 1.0), -1.0)
        
    q1 = math.acos(u)
    q2 = math.atan2(y, x) - math.atan2(L2*np.sin(q1), L1+L2*np.cos(q1))
    
    if q1 < 0:
        q1 = q1 + 2*math.pi

    return np.array([q2, q1])

    
# --- Cost Functions (from Table I in the paper) ---
def cost_functions(s, ω):
    """
    s: State trajectory (n_steps × 6)
    u: Control inputs (n_steps × 2)
    ω: Weight vector for cost functions
    Returns: Total cost (scalar)
    """
    C = features_vector(s)
    print('cost :', np.dot(ω, C))
 
    return np.dot(ω, C)

def features_vector(s):
    s = s.reshape(-1, 6)
    θ= s[:, 0:2].T
    dθ = s[:, 2:4].T
    ddθ = s[:, 4:6].T
    
    X = direct_kinematics(s[:, 0:2].T, l1, l2)
  

    dX = np.zeros_like(X)
    # analytic end-effector velocities using Jacobian * q_dot
    dX[0,:] = -l1 * np.sin(θ[0,:]) * dθ[0,:] - l2 * np.sin(θ[0,:] + θ[1,:]) * (dθ[0,:] + dθ[1,:])
    dX[1,:] =  l1 * np.cos(θ[0,:]) * dθ[0,:] + l2 * np.cos(θ[0,:] + θ[1,:]) * (dθ[0,:] + dθ[1,:])
    
    τ = np.zeros_like(θ)
    for i in range(len(s)):
        τ[:,i] = M_cached(θ[0][i], θ[1][i]) @ np.array([ddθ[0][i], ddθ[1][i]]).T + R_cached(θ[0][i], θ[1][i], dθ[0][i], dθ[1][i]) @ np.array([dθ[0][i], dθ[1][i]]).T + g_vec_cached(θ[0][i], θ[1][i]).T
    # Cost terms (normalized as in Table I)
    C = np.zeros((5, 2))
    C[0] = np.sum(dθ.T @ dθ)  # Minimum joint vel
    C[1] = np.sum(τ.T @ τ)  # Minimum joint torque
    C[2] = np.sum(dX.T @ dX)  # minimum end effector vel
    C[3] = np.sum(np.gradient(ddθ, axis=0).T @ np.gradient(ddθ, axis=0))  # joints jerk
    C[4] = np.sum(np.gradient(τ, axis=0).T @ np.gradient(τ, axis=0))  # torque change
    return np.sum(C, axis=1)  # sum over time steps


# --- Constraint functions ---
def start_con_fun(x, n_steps):
    s=x.reshape(n_steps,6)
    return (s[0] - start).ravel()

def goal_con_fun(x, n_steps):
    s=x.reshape(n_steps,6)
    return (s[-1] - goal).ravel()

def vel_con_fun(x, n_steps, dt):
    s=x.reshape(n_steps,6)
    X = s[:, :2]     # angles
    dX = s[:, 2:4]   # angular velocities
    cons = []
    for i in range(n_steps-1):
        cons.append((X[i+1] - X[i] - dt * dX[i]).ravel())
    return np.concatenate(cons)

def accel_con_fun(x, n_steps, dt):
    s=x.reshape(n_steps,6)
    dX = s[:, 2:4]   # velocities
    ddX = s[:, 4:6]  # accelerations
    cons = []
    for i in range(n_steps-1):
        cons.append((dX[i+1] - dX[i] - dt * ddX[i]).ravel())
    return np.concatenate(cons)



# --- DOC Solver ---
def solve_DOC(n_steps=20, dt=0.1, w_ref=None):
    if w_ref is None:
        w_ref = np.array([0.981, 0.196, 0.002, 0.010, 0, 0, 0, 0])
        
    # initial guess: linear interpolation for angles, zero velocities/accel
    S0 = np.zeros((n_steps,6))
    U0 = np.zeros((n_steps,2))
    x0 = np.hstack((S0.flatten()))
    
    # bounds: states unbounded, torques unbounded
    bounds = [(None, None), (None, None), (None, None), (None, None), (-200, 200), (-200, 200)]*(n_steps)

    # constraints list
    cons = [
        {'type':'eq', 'fun': lambda x, ns=n_steps: start_con_fun(x, ns)},
        {'type':'eq', 'fun': lambda x, ns=n_steps: goal_con_fun(x, ns)},
        {'type':'eq', 'fun': lambda x, ns=n_steps, dt=dt: vel_con_fun(x, ns, dt)},
        {'type':'eq', 'fun': lambda x, ns=n_steps, dt=dt: accel_con_fun(x, ns, dt)},
    ]
    
    def objective(x):
        s=x.reshape(n_steps,6)
        return cost_functions(s, w_ref)

    res = minimize(objective, x0, method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter':500, 'ftol':1e-6, 'disp': True})

    S_opt = res.x.reshape(n_steps,6)
    θ1, θ2 = S_opt[:, 0], S_opt[:, 1]
    dθ1, dθ2 = S_opt[:, 2], S_opt[:, 3]
    ddθ1, ddθ2 = S_opt[:, 4], S_opt[:, 5]
    τ1, τ2 = np.zeros(n_steps), np.zeros(n_steps)
    for i in range(n_steps):
        τ1[i], τ2[i] = M_cached(θ1[i], θ2[i]) @ np.array([ddθ1[i], ddθ2[i]]).T + R_cached(θ1[i], θ2[i], dθ1[i], dθ2[i]) @ np.array([dθ1[i], dθ2[i]]).T + g_vec_cached(θ1[i], θ2[i]).T

    U_opt = np.vstack((τ1, τ2)).T
    return S_opt , U_opt, res


def h_constraints(x, n_steps=n_steps, dt=dt):
    """Combine all equality constraints into one vector h(z)."""
    return np.concatenate([start_con_fun(x, n_steps), goal_con_fun(x, n_steps), vel_con_fun(x, n_steps, dt), accel_con_fun(x, n_steps, dt)])

def callbackFunc(Xi):    
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2]))
    Nfeval += 1
    
    
def KKT_vector(x, nu, w):
    
    """Compute KKT residuals K(z, nu; theta) = [∂zL; h(z)]"""
    # numerical gradient of f wrt z
    eps = 1e-6
    grad_f = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        grad_f[i] = (cost_functions(x+e, w) - cost_functions(x-e, w)) / (2*eps)

    # constraint Jacobian J_h (finite difference)
    h0 = h_constraints(x)
    m = len(h0)
    J_h = np.zeros((m, len(x)))
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        J_h[:,i] = (h_constraints(x+e) - h_constraints(x-e)) / (2*eps)

    # KKT stationarity and feasibility
    stationarity = grad_f + J_h.T @ nu
    feasibility = h0
    return np.concatenate([stationarity, feasibility])

def single_level_ioc(S_obs, w0=None):
    """
    Implements Algorithm 2 from Becanovic et al. (2024)
    Estimate theta that explains observed trajectory y_obs
    """
    if w0 is None:
        w0 = np.abs(np.random.rand(5))
        w0 /= np.sum(w0)

    # S contains q, qd, qdd
    x0 = S_obs.flatten()
    print('h_constraints(x0):', h_constraints(x0))
    stop()
    nu0 = np.zeros(len(h_constraints(x0)))  # initialize multipliers

    z_init = np.concatenate([x0, nu0, w0])
    print('Initialization end')

    
    def objective_ioc(z):
        x = z[:len(x0)]
        return 0.5 * np.linalg.norm(x - S_obs.flatten())**2
    
    def constraint_fun(z):
        x = z[:len(x0)]
        nu = z[len(x0):len(x0)+len(h_constraints(x0))]
        theta = z[-5:]
        return KKT_vector(x, nu, theta)

    cons = {'type':'eq', 'fun': constraint_fun}
    
    print('Starting minimization')
    res = minimize(objective_ioc, z_init, method='SLSQP', constraints=cons,
                   callback=callbackFunc,
                   options={'maxiter':10, 'ftol':1e-3, 'disp':True})

    # extract results
    x_est = res.x[:len(x0)]
    nu_est = res.x[len(x0):len(x0)+len(h_constraints(x0))]
    theta_est = res.x[-5:]
    theta_est = np.maximum(theta_est, 0)
    theta_est /= np.sum(theta_est)

    return theta_est, x_est, res


def save_image(filename):
  
    # PdfPages is a wrapper around pdf 
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(ROOT + '/' + filename)
    
    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    
    # iterating over the numbers in list
    for fig in figs: 
      
        # and saving the files
        fig.savefig(p, format='pdf') 
        
    # close the object
    p.close()  
    
    

# --- Example run ---
if __name__ == "__main__":

    w_ref = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 3, 0, 0],
                      [0, 0, 0, 0, 1]])
              

    for w in w_ref:
        w = w / np.sum(w)
        
        # # # Generate trajectories with direct optimal control
        # S_opt, U_opt, res = solve_DOC(n_steps=n_steps, dt=dt, w_ref=w)
        # np.savetxt(f"S_opt_{np.round(w,3)}.csv", S_opt, delimiter=",")
        # np.savetxt(f"U_opt_{np.round(w,3)}.csv", U_opt, delimiter=",")
        # print("Success:", res.success, "| Cost:", res.fun)
        # print("Start state:", S_opt[0])
        # print("End state:", S_opt[-1])
        # print("Sample torques:\n", U_opt[-5:])
        
        
        # fig = plt.figure(figsize=(8,4))
        # plt.plot(S_opt[:,0], label='θ1')
        # plt.plot(S_opt[:,1], label='θ2')
        # plt.xlabel('Time step')
        # plt.ylabel('Angle [rad]')
        # plt.legend()
        # plt.title('Joint angles (optimized) with w=' + np.array2string(w, precision=3, separator=','))
        
        # plt.figure(figsize=(8,4))
        # plt.plot(S_opt[:,2], label='dθ1/dt')
        # plt.plot(S_opt[:,3], label='dθ2/dt')
        # plt.xlabel('Time step')
        # plt.ylabel('Velocity [rad/s]')
        # plt.legend()
        # plt.title('Joint velocities (optimized)')

        # plt.figure(figsize=(8,4))
        # plt.plot(S_opt[:,4], label='ddθ1/dt²')
        # plt.plot(S_opt[:,5], label='ddθ2/dt²')
        # plt.xlabel('Time step')
        # plt.ylabel('Acceleration [rad/s²]')
        # plt.legend()
        # plt.title('Joint accelerations (optimized)with w=' + np.array2string(w, precision=3, separator=','))

        # plt.figure(figsize=(8,4))
        # plt.plot(U_opt[:,0], label='τ1')
        # plt.plot(U_opt[:,1], label='τ2')
        # plt.xlabel('Time step')
        # plt.ylabel('Torque [Nm]')
        # plt.legend()
        # plt.title('Torques (optimized)')
        
        # hand_traj = np.array([direct_kinematics(S_opt[i,0:2], l1, l2) for i in range(n_steps)])
        
        # plt.figure(figsize=(6,6))
        # plt.plot(hand_traj[:,0], hand_traj[:,1], '-')
        # plt.plot(direct_kinematics(start[0:2], l1, l2)[0], direct_kinematics(start[0:2], l1, l2)[1], 'go', label='Start')
        # plt.plot(direct_kinematics(goal[0:2], l1, l2)[0], direct_kinematics(goal[0:2], l1, l2)[1], 'ro', label='Goal')
        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-0.5, 2.5)
        # plt.xlabel('X [m]')
        # plt.ylabel('Y [m]')
        # plt.title('Hand trajectory (optimized) with w=' + np.array2string(w, precision=3, separator=','))
        # plt.legend() 
        # plt.show()
        
        #Load S_opt from the corresponding CSV file
        S_opt = np.loadtxt(f"S_opt_{np.round(w,3)}.csv", delimiter=",")
        print("Loaded S_opt shape:", S_opt.shape)
        U_opt = np.loadtxt(f"U_opt_{np.round(w,3)}.csv", delimiter=",")
        print("Loaded U_opt shape:", U_opt.shape)
        
        
        print("IOC start")
        # Step 2: estimate theta using single-level IOC
        w_est, x_est, res = single_level_ioc(S_opt)
        
        print("Estimated w:", w_est)
        print('True w:', w)
        print("Success:", res.success, "| Cost:", res.fun)
        print("IOC end")
        


        
