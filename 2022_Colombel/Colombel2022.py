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

# Intermediate terms for M(θ) and G(θ, dθ)
a1 = m1*r1**2 + m2*(l1**2 + r2**2) + I1 + I2
a2 = m2*l1*r2
a3 = m2*r2**2 + I2

# Intermediate terms for g(θ)
b1 = l1*m2 + r1*m1
b2 = r2*m2

# Goal state: [π/2, π/2, 0, 0, 0, 0]
goal = np.array([np.pi/2, -np.pi/2, 0, 0, 0, 0])
# Initial state: [0, 0, 0, 0, 0, 0]
start = np.array([0, 0, 0, 0, 0, 0])

dt = 0.1  # Time step

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
    return np.dot(ω, C)

def features_vector(s):
    s = s.reshape(-1, 6)
    θ1, θ2 = s[:, 0], s[:, 1]
    dθ1, dθ2 = s[:, 2], s[:, 3]
    ddθ1, ddθ2 = s[:, 4], s[:, 5]  
    
    τ1, τ2 = np.zeros_like(θ1), np.zeros_like(θ2)
    for i in range(len(s)):
        τ1[i], τ2[i] = M_cached(θ1[i], θ2[i]) @ np.array([ddθ1[i], ddθ2[i]]).T + R_cached(θ1[i], θ2[i], dθ1[i], dθ2[i]) @ np.array([dθ1[i], dθ2[i]]).T + g_vec_cached(θ1[i], θ2[i]).T
  
    # Cost terms (normalized as in Table I)
    C = np.zeros(8)
    C[0] = np.sum(τ1**2)  # τ₁²
    C[1] = np.sum(τ2**2)  # τ₂²
    C[2] = np.sum(np.gradient(ddθ1, axis=0)**2)  # jerk₁²
    C[3] = np.sum(np.gradient(ddθ2, axis=0)**2)  # jerk₂²
    C[4] = np.sum(ddθ1**2)  # ÿ₁²
    C[5] = np.sum(ddθ2**2)  # ÿ₂²
    C[6] = np.sum((dθ1 * τ1)**2)  # (dθ₁τ₁)²
    C[7] = np.sum((dθ2 * τ2)**2)  # (dθ₂τ₂)²
    return C


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




# --- main analytic Jacobian builder ---
def analytic_J_omega(S, dt):
    """
    S: (n_steps, 6) states (theta1,theta2,dtheta1,dtheta2,ddtheta1,ddtheta2)
    dt: timestep for jerk finite-difference
    Returns J_omega: shape (ns, 8) where ns = 6*n_steps
    Each column k corresponds to ∂C_k / ∂s (flattened).
    """
    S = np.asarray(S).reshape(-1, 6)
    n_steps = S.shape[0]
    ns = 6 * n_steps
    nc = 8
    J = np.zeros((ns, nc))  # output

    # helper to index flattened state: for time t, the 6 indices are base..base+5
    def base_idx(t):
        return 6 * t

    # Precompute local taus and local partials ∂τ/∂s_t (2x6) for each time
    taus = np.zeros((n_steps, 2))
    dtaudst = np.zeros((n_steps, 2, 6))  # for each t: 2x6 matrix flattened as [row, state_index]

    for t in range(n_steps):
        th1, th2, d1, d2, dd1, dd2 = S[t]
        theta = np.array([th1, th2])
        dtheta = np.array([d1, d2])
        ddtheta = np.array([dd1, dd2])

        # compute tau at this time
        Mloc = M_cached(th1, th2)
        Rloc = R_cached(th1, th2, d1, d2)
        gloc = g_vec_cached(th1, th2)
        tau = Mloc @ ddtheta + Rloc @ dtheta + gloc
        taus[t, :] = tau

        # derivatives
        # ∂τ/∂ddtheta = M
        dtaud_dd = Mloc.copy()  # 2x2

        # ∂τ/∂dtheta = R + sum_j (∂R/∂d_j) * d_j
        # compute ∂R/∂d1, ∂R/∂d2
        dR_d1 = dR_dd1(th2)
        dR_d2 = dR_dd2(th2)
        dtaud_d = Rloc + dR_d1 * d1 + dR_d2 * d2  # 2x2 (numpy broadcasts scalars properly)

        # ∂τ/∂theta
        # ∂M/∂θ2 * ddtheta contributes to theta2 column, M independent of theta1
        dMth2 = dM_dtheta2(th2)
        # ∂R/∂θ2 * dtheta contributes to theta2 column
        dRth2 = dR_dtheta2(th2, d1, d2)
        dgth1, dgth2 = dg_dtheta(th1, th2)

        # assemble 2x6 derivative: columns correspond to [θ1, θ2, d1, d2, dd1, dd2]
        dtaud_s = np.zeros((2,6))

        # θ1 column:
        # ∂τ/∂θ1 = dg/dθ1 (M and R do not depend on θ1 in your simplified model)
        dtaud_s[:,0] = dgth1

        # θ2 column:
        dtaud_s[:,1] = (dMth2 @ ddtheta) + (dRth2 @ dtheta) + dgth2

        # d1 and d2 columns (velocities)
        dtaud_s[:,2:4] = dtaud_d  # ∂τ/∂d1, ∂τ/∂d2 placed as two columns

        # dd1 and dd2 columns (accelerations)
        dtaud_s[:,4:6] = dtaud_dd  # ∂τ/∂dd1, ∂τ/∂dd2

        dtaudst[t,:,:] = dtaud_s

    # --- Now compute each feature's contribution across time ---

    # Feature 0: sum_t tau1(t)^2
    # Feature 1: sum_t tau2(t)^2
    for t in range(n_steps):
        base = base_idx(t)
        tau1 = taus[t,0]; tau2 = taus[t,1]
        dtaud_s = dtaudst[t]  # 2 x 6

        # ∂(tau1^2)/∂s_t = 2 * tau1 * ∂tau1/∂s_t  (tau1 is first row)
        J[base:base+6, 0] += 2.0 * tau1 * dtaud_s[0, :]

        # ∂(tau2^2)/∂s_t
        J[base:base+6, 1] += 2.0 * tau2 * dtaud_s[1, :]

    # Feature 2 and 3: jerk1^2 and jerk2^2
    # jerk_i = (ddθ[t+1] - ddθ[t]) / dt for t=0..n-2
    # C2 = sum_{t=0}^{n-2} jerk1(t)^2
    # For each dd1[k], contributions from jerk at k-1 and k
    # We'll compute j1 array and then its contributions
    if n_steps >= 2:
        j1 = np.zeros(n_steps-1)
        j2 = np.zeros(n_steps-1)
        for i in range(n_steps-1):
            j1[i] = (S[i+1,4] - S[i,4]) / dt
            j2[i] = (S[i+1,5] - S[i,5]) / dt

        # For ddtheta1 (state index 4)
        for k in range(n_steps):
            base = base_idx(k)
            # derivative wrt dd1[k]
            contrib = np.zeros(6)
            # from j_k (if k <= n-2): j_k = (dd1[k+1] - dd1[k])/dt -> d j_k / d dd1[k] = -1/dt
            if k <= n_steps - 2:
                contrib[4] += 2.0 * j1[k] * (-1.0/dt)
            # from j_{k-1} (if k >= 1): j_{k-1} = (dd1[k] - dd1[k-1])/dt -> d j_{k-1} / d dd1[k] = +1/dt
            if k - 1 >= 0:
                contrib[4] += 2.0 * j1[k-1] * (1.0/dt)
            # Add to column 2 (jerk1^2)
            J[base:base+6, 2] += contrib

            # Similarly for dd2 (state index 5) and feature 3
            contrib2 = np.zeros(6)
            if k <= n_steps - 2:
                contrib2[5] += 2.0 * j2[k] * (-1.0/dt)
            if k - 1 >= 0:
                contrib2[5] += 2.0 * j2[k-1] * (1.0/dt)
            J[base:base+6, 3] += contrib2
    else:
        # n_steps == 1 => no jerk terms (they are zero and have zero derivative)
        pass

    # Feature 4 and 5: ddtheta1^2, ddtheta2^2  (these only depend on dd components at same t)
    for t in range(n_steps):
        base = base_idx(t)
        dd1 = S[t,4]; dd2 = S[t,5]
        # ∂(dd1^2)/∂s_t = 2*dd1 * ∂dd1/∂s_t -> only at index 4
        vec4 = np.zeros(6); vec4[4] = 2.0 * dd1
        vec5 = np.zeros(6); vec5[5] = 2.0 * dd2
        J[base:base+6, 4] += vec4
        J[base:base+6, 5] += vec5

    # Feature 6 and 7: (tau1*d1)^2 and (tau2*d2)^2
    for t in range(n_steps):
        base = base_idx(t)
        tau1 = taus[t,0]; tau2 = taus[t,1]
        d1 = S[t,2]; d2 = S[t,3]
        dtaud_s = dtaudst[t]  # 2 x 6

        # For f = (tau1 * d1)^2
        # df/ds = 2*(tau1*d1) * ( d1 * ∂tau1/∂s + tau1 * ∂d1/∂s )
        factor1 = 2.0 * (tau1 * d1)
        vec_tau1 = dtaud_s[0, :]  # ∂tau1/∂s (6,)
        e_d1 = np.zeros(6); e_d1[2] = 1.0
        J[base:base+6, 6] += factor1 * (d1 * vec_tau1 + tau1 * e_d1)

        # For f = (tau2 * d2)^2
        factor2 = 2.0 * (tau2 * d2)
        vec_tau2 = dtaud_s[1, :]
        e_d2 = np.zeros(6); e_d2[3] = 1.0
        J[base:base+6, 7] += factor2 * (d2 * vec_tau2 + tau2 * e_d2)

    return J

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


def solve_kkt_ioc(S_opt, U_opt, dt, start, goal, w0=None, eps=1e-6, svd_drop_threshold_ratio=1e2):
    """
    Solve IOC using KKT stationary conditions + SVD nullspace method (as in Colombel et al. ICRA 2022).
    Inputs:
      S_opt : (n_steps x 6) observed trajectory (state vector xt = [θ1,θ2,dθ1,dθ2,ddθ1,ddθ2])
      U_opt : (n_steps x 2) observed torques (not used directly here but kept for API)
      dt    : timestep used in dynamics constraints
      start, goal : arrays of length 6 (start/goal states)
      w0    : optional initial guess for ω (not required)
      eps   : finite-difference step for ∂Ck/∂s
      svd_drop_threshold_ratio : minimum singular-value ratio considered a "drop" (default 1e2)
    Returns:
      w_est : estimated normalized weight vector (nc=8)
      res   : dict with fields:
              - 'z' : stacked vector [ω; λ] (length nc + nf)
              - 'omega_raw' : ω as extracted from SVD right-singular vector (before positivity projection)
              - 'lambda' : estimated λ (from least squares given ω)
              - 'singular_values' : singular values of J
              - 'drop' : detected rank drop (integer)
              - 'drop_index' : index of largest ratio
              - 'residual_norm' : ||J z||_2
              - 'J' : the KKT matrix used (normalized)
              - 'success' : boolean (True if drop==1 and ω meets positivity after projection)
    Notes:
      - The function constructs J = [J_omega, J_lambda] where J_omega columns are ∂Ck/∂s
        and J_lambda columns are analytic gradients of the equality constraints.
      - The cost feature functions used must match your features_vector(s) implementation
        (here we call that function).
    """
    S = np.asarray(S_opt).reshape(-1, 6)
    n_steps = S.shape[0]
    ns = 6 * n_steps
    nc = 8  # number of cost terms used in features_vector 
    
    # Build J_lambda analytically (constraints are linear in s):
    # nf = 6 (start) + 6 (goal) + 2*(n_steps-1) (vel) + 2*(n_steps-1) (accel)
    nf = 12 + 4 * (n_steps - 1)
    J_lambda = np.zeros((ns, nf))

    # helper to set block in J_lambda: state index t, variable block offsets:
    # state ordering in flatten: for t in [0..n_steps-1], block indices base = 6*t .. 6*t+5
    def base_idx(t):
        return 6 * t

    # start constraint: s0 - start = 0
    J_lambda[base_idx(0):base_idx(0) + 6, 0:6] = np.eye(6)

    # goal constraint: sN - goal = 0
    J_lambda[base_idx(n_steps - 1):base_idx(n_steps - 1) + 6, 6:12] = np.eye(6)

    # velocity constraints:
    # for i in 0..n_steps-2: X[i+1] - X[i] - dt * dX[i] = 0  (X = angles, indices 0:2, dX = 2:4)
    vel_start_col = 12
    col = vel_start_col
    for i in range(n_steps - 1):
        bi = base_idx(i)
        bnext = base_idx(i + 1)
        # derivative wrt X[i+1] (θ) => +I (2x2) placed at indices [bnext:bnext+2]
        J_lambda[bnext + 0:bnext + 2, col:col + 2] = np.eye(2)
        # derivative wrt X[i] => -I
        J_lambda[bi + 0:bi + 2, col:col + 2] = -np.eye(2)
        # derivative wrt dX[i] => -dt*I (dX indices in state are 2:4)
        J_lambda[bi + 2:bi + 4, col:col + 2] = -dt * np.eye(2)
        col += 2

    # accel constraints:
    # for i in 0..n_steps-2: dX[i+1] - dX[i] - dt * ddX[i] = 0  (dX indices 2:4, ddX 4:6)
    accel_start_col = col
    for i in range(n_steps - 1):
        bi = base_idx(i)
        bnext = base_idx(i + 1)
        # derivative wrt dX[i+1] => +I
        J_lambda[bnext + 2:bnext + 4, col:col + 2] = np.eye(2)
        # derivative wrt dX[i] => -I
        J_lambda[bi + 2:bi + 4, col:col + 2] = -np.eye(2)
        # derivative wrt ddX[i] => -dt * I (indices 4:6)
        J_lambda[bi + 4:bi + 6, col:col + 2] = -dt * np.eye(2)
        col += 2

    # Sanity check column count
    assert col == nf, f"constructed {col} cols but expected nf={nf}"

    # Build J_omega via finite difference of each cost feature Ck(s):
    # features_vector expects flattened s (n_steps*6) and returns vector length nc
    s_flat = S.flatten()
    def eval_C_vector(sflat):
        # features_vector expects s shaped (-1,6)
        return features_vector(np.asarray(sflat).reshape(-1, 6))

    # baseline
    C0 = eval_C_vector(s_flat)  # shape (nc,)
    J_omega = np.zeros((ns, nc))
    # central finite differences
    # for j in range(ns):
    #     e = np.zeros_like(s_flat)
    #     e[j] = eps
    #     Cp = eval_C_vector(s_flat + e)
    #     Cm = eval_C_vector(s_flat - e)
    #     J_omega[j, :] = (Cp - Cm) / (2 * eps)
    J_omega=analytic_J_omega(S, dt)

    # Build full J = [J_omega, J_lambda] (ns x (nc + nf))
    J = np.hstack([J_omega, J_lambda])

    # Normalize J as in paper: matrix normalized by maximum absolute column sum (the paper uses
    # "max absolute column sum = 1" normalization). We compute the max abs element and divide.
    max_abs = np.max(np.abs(J))
    if max_abs > 0:
        J_norm = J / max_abs
    else:
        J_norm = J.copy()

    # SVD
    U, svals, Vt = svd(J_norm, full_matrices=False)
    V = Vt.T
    L = len(svals)
    # compute successive ratios to detect drop
    ratios = svals[:-1] / (svals[1:] + 1e-20)
    drop_idx = int(np.argmax(ratios))
    drop = L - (drop_idx + 1)  # as per paper: if drop==1 -> single zero singular value at the end

    # Decide which singular vectors we consider "null"
    # We use an heuristic threshold: largest ratio must exceed svd_drop_threshold_ratio
    largest_ratio = ratios[drop_idx] if ratios.size > 0 else 0.0
    use_svd_solution = (largest_ratio >= svd_drop_threshold_ratio) and (drop >= 1)

    result = {}
    result['singular_values'] = svals
    result['drop'] = drop
    result['drop_index'] = drop_idx
    result['largest_ratio'] = largest_ratio
    result['J'] = J_norm

    if (not use_svd_solution):
        # Not reliable (full rank or no clear drop) -> return diagnostics and a failure flag
        result['z'] = None
        result['omega_raw'] = None
        result['lambda'] = None
        result['residual_norm'] = np.linalg.norm(J_norm @ np.zeros(nc + nf))
        result['success'] = False
        return None, result

    # If drop >= 1, the last 'drop' columns of V are in nullspace approximate.
    # For the simple case drop==1, we choose last singular vector V[:, -1].
    # If drop>1, we choose last singular vector as suggested by the paper (they pick VL).
    z_raw = V[:, -1]  # right singular vector associated with smallest singular value
    omega_raw = z_raw[:nc].copy()
    lambda_raw = z_raw[nc:].copy()

    # Normalize omega_raw to unit 2-norm (as in paper they renormalize)
    if np.linalg.norm(omega_raw) > 0:
        omega_normed = omega_raw / np.linalg.norm(omega_raw)
    else:
        omega_normed = omega_raw.copy()

    # The paper requires ω >= 0 (up to a global sign). If the vector is mostly negative, flip sign.
    if np.sum(omega_normed < 0) > np.sum(omega_normed > 0):
        omega_normed = -omega_normed

    # If any negative entries remain, the paper says prefer not to provide a solution.
    # We will project negatives to zero and renormalize (practical compromise).
    omega_proj = omega_normed.copy()
    if np.any(omega_proj < 0):
        omega_proj[omega_proj < 0] = 0.0
        if np.linalg.norm(omega_proj) > 0:
            omega_proj = omega_proj / np.linalg.norm(omega_proj)
        else:
            # projection eliminated everything => unreliable
            result['z'] = z_raw
            result['omega_raw'] = omega_raw
            result['lambda'] = lambda_raw
            result['residual_norm'] = np.linalg.norm(J_norm @ z_raw)
            result['success'] = False
            return None, result

    # Given ω fixed (omega_proj), solve for λ minimizing || J_omega * ω + J_lambda * λ || (linear least squares)
    b = - (J_omega @ omega_proj)  # target is -J_omega * ω
    # Solve J_lambda * λ = b (least squares)
    # Use np.linalg.lstsq for robust solution
    if J_lambda.size == 0:
        lambda_est, *_ = (np.zeros(0),)
    else:
        lambda_est, *_ = lstsq(J_lambda, b, rcond=None)

    # Build stacked z and compute residual norm
    z_est = np.hstack([omega_proj, lambda_est])
    residual_norm = np.linalg.norm(J_norm @ z_est)

    # Pack results
    result['z'] = z_est
    result['omega_raw'] = omega_raw
    result['omega_proj'] = omega_proj
    result['lambda'] = lambda_est
    result['residual_norm'] = residual_norm
    result['success'] = True

    return omega_proj, result

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
    n_steps = 35
    dt = 0.01
    w_ref = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0],
                      [0.9, 0.1, 0, 0, 0, 0, 0, 0],
                      [0.1, 0.9, 0, 0, 0, 0, 0, 0],
                      [0.981, 0.196, 0, 0, 0, 0, 0, 0],
                      [0.196, 0.981, 0, 0, 0, 0, 0, 0],
                      [0.196, 0.918, 0.002, 0.01, 0, 0, 0, 0],
                      [0.117, 0.078, 0, 0, 0.971, 0.194, 0, 0],
                      [0.0002, 0.001, 0.002, 0.01, 0.981, 0.196, 0, 0], 
                      [0.004, 0.004, 0.007, 0.007, 0.707, 0.707, 0, 0],
                      [0.019, 0.097, 0.002, 0.010, 0.971, 0.194, 0.019, 0.097]])  # Example weights

    for w in w_ref:
        w = w / np.sum(w)
        
        # # Generate trajectories with direct optimal control
        S_opt, U_opt, res = solve_DOC(n_steps=n_steps, dt=dt, w_ref=w)
        # np.savetxt(f"S_opt_{np.round(w,3)}.csv", S_opt, delimiter=",")
        # np.savetxt(f"U_opt_{np.round(w,3)}.csv", U_opt, delimiter=",")
        print("Success:", res.success, "| Cost:", res.fun)
        print("Start state:", S_opt[0])
        print("End state:", S_opt[-1])
        print("Sample torques:\n", U_opt[-5:])
        
        
        fig = plt.figure(figsize=(8,4))
        plt.plot(S_opt[:,0], label='θ1')
        plt.plot(S_opt[:,1], label='θ2')
        plt.xlabel('Time step')
        plt.ylabel('Angle [rad]')
        plt.legend()
        plt.title('Joint angles (optimized) with w=' + np.array2string(w, precision=3, separator=','))
        
        plt.figure(figsize=(8,4))
        plt.plot(S_opt[:,2], label='dθ1/dt')
        plt.plot(S_opt[:,3], label='dθ2/dt')
        plt.xlabel('Time step')
        plt.ylabel('Velocity [rad/s]')
        plt.legend()
        plt.title('Joint velocities (optimized)')

        plt.figure(figsize=(8,4))
        plt.plot(S_opt[:,4], label='ddθ1/dt²')
        plt.plot(S_opt[:,5], label='ddθ2/dt²')
        plt.xlabel('Time step')
        plt.ylabel('Acceleration [rad/s²]')
        plt.legend()
        plt.title('Joint accelerations (optimized)with w=' + np.array2string(w, precision=3, separator=','))

        plt.figure(figsize=(8,4))
        plt.plot(U_opt[:,0], label='τ1')
        plt.plot(U_opt[:,1], label='τ2')
        plt.xlabel('Time step')
        plt.ylabel('Torque [Nm]')
        plt.legend()
        plt.title('Torques (optimized)')
        
        hand_traj = np.array([direct_kinematics(S_opt[i,0:2], l1, l2) for i in range(n_steps)])
        
        plt.figure(figsize=(6,6))
        plt.plot(hand_traj[:,0], hand_traj[:,1], '-')
        plt.plot(direct_kinematics(start[0:2], l1, l2)[0], direct_kinematics(start[0:2], l1, l2)[1], 'go', label='Start')
        plt.plot(direct_kinematics(goal[0:2], l1, l2)[0], direct_kinematics(goal[0:2], l1, l2)[1], 'ro', label='Goal')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.5, 2.5)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Hand trajectory (optimized) with w=' + np.array2string(w, precision=3, separator=','))
        plt.legend() 
        
        #Load S_opt from the corresponding CSV file
        # S_opt = np.loadtxt(f"S_opt_{np.round(w,3)}.csv", delimiter=",")
        # print("Loaded S_opt shape:", S_opt.shape)
        # U_opt = np.loadtxt(f"U_opt_{np.round(w,3)}.csv", delimiter=",")
        # print("Loaded U_opt shape:", U_opt.shape)
        
        w_est, res = solve_kkt_ioc(S_opt, U_opt, dt, start, goal, w0=w)
        
        if w_est is not None:
            w_est = w_est * (np.sum(w) / np.sum(w_est))
            
        print("KKT-IOC success:", res['success'])
        print("Estimated w:", w_est)
        print("Residual norm:", res['residual_norm'])
        print("Singular values (smallest first):", res['singular_values'][-8:])
        print("Detected rank drop:", res['drop'], "largest ratio:", res['largest_ratio'])

       
        
        plt.figure(figsize=(8,4))
        plt.bar(np.arange(len(w)), w, alpha=0.6, label='True w')
        plt.bar(np.arange(len(w)), w_est, alpha=0.6, label='Estimated w')
        plt.plot(np.arange(len(w)), w - w_est, 'ko--', label='w - w_est')
        plt.xlabel('Feature index')
        plt.ylabel('Weight')
        plt.xticks(np.arange(8), ['tau1', 'tau2', 'jerk1', 'jerk2', 'accel1', 'accel2', 'power1', 'power2'])
        plt.legend()
        plt.title('True vs Estimated weights (w - w_est)')
        
        plt.show()
      

        
    save_image("optimized_results.pdf")