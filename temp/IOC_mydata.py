# Inverse Optimal Control for discrete-time LQR (user-visible demo)
# - This notebook provides functions to:
#   1) estimate linear dynamics A,B from (x,u) trajectories
#   2) estimate empirical feedback K from u = -K x
#   3) recover diagonal Q and scalar R by minimizing difference between estimated K and LQR K(Q,R)
# - The code is general: if you have your own dataset, replace the synthetic data generation
#   with your arrays `X` (states) and `U` (controls).
#
# Outputs: prints estimated A,B,K, recovered Q,R and shows a small plot comparing K entries.
#
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
from scipy.signal import savgol_filter
from numpy.linalg import pinv
from temp.FourDofArm import *


# ---------- finite-horizon LQR solver for time-varying discrete linear system ----------
def finite_horizon_LQR(A_list, B_list, Q, R, Qf, x0):
    """
    Given A_list, B_list (length N), returns u_pred (N-1 x m) and x_pred (N x state_dim)
    Q,R,Qf are matrices for cost. x0 initial state.
    We solve backward Riccati and forward simulate u = -K_k x.
    """
    N = len(A_list)
    n = A_list[0].shape[0]
    m = B_list[0].shape[1]

    # backward pass: compute P_k, K_k for k = N-1...0
    P = [None]*(N+1)
    K = [None]*N
    P[N] = Qf.copy()
    for k in range(N-1, -1, -1):
        A = A_list[k]
        B = B_list[k]
        # S = R + B^T P_{k+1} B
        S = R + B.T @ P[k+1] @ B
        # K = S^{-1} B^T P_{k+1} A
        Kk = np.linalg.solve(S, B.T @ P[k+1] @ A)
        K[k] = Kk
        # Riccati update
        P[k] = Q + A.T @ P[k+1] @ (A - B @ Kk)

    # forward simulate using observed x0 (we use closed-loop u = -K x)
    x = np.zeros((N, n))
    u = np.zeros((N, m))
    x[0,:] = x0
    for k in range(N-1):
        u[k,:] = -K[k] @ x[k,:]
        x[k+1,:] = A_list[k] @ x[k,:] + B_list[k] @ u[k,:]
    # last control can be -K[N-1] x[N-1]
    u[N-1,:] = -K[N-1] @ x[N-1,:]
    return x, u, K, P

# ---------- parameterization of Q,R,Qf to ensure PSD / PD ----------
def param_to_QR(params, state_dim, control_dim):
    """
    params: vector containing lower-triangular entries of L for Q = L L^T (state_dim x state_dim)
            then log-diagonal entries for R (control_dim)
            and optionally Qf lower-tri entries for Qf = Lf Lf^T
    We'll assume Qf = Q for simplicity unless params includes extra entries.
    """
    # indices
    n = state_dim
    m = control_dim
    # number of lower-tri entries
    lt = n*(n+1)//2
    L_entries = params[:lt]
    # form lower triangular L
    L = np.zeros((n,n))
    idx = 0
    for i in range(n):
        for j in range(i+1):
            L[i,j] = L_entries[idx]; idx += 1
    Q = L @ L.T
    # R diag entries
    log_r = params[lt:lt+m]
    R = np.diag(np.exp(log_r))  # positive diagonal
    # set Qf = Q (or you could parameterize Qf too)
    Qf = Q.copy()
    return Q, R, Qf

# ---------- loss function: how well predicted u matches observed tau ----------
def ioc_loss(params, A_list, B_list, X, U, n_state, n_ctrl):
    Q, R, Qf = param_to_QR(params, n_state, n_ctrl)
    x0 = X[0,:]
    _, u_pred, _, _ = finite_horizon_LQR(A_list, B_list, Q, R, Qf, x0)
    # loss: squared error between observed U and predicted u_pred (optionally ignore first few frames)
    loss = np.sum((U - u_pred)**2)
    # some regularization to keep Q small
    loss += 1e-6 * np.sum(np.diag(Q)**2) + 1e-6 * np.sum(np.diag(R)**2)
    return loss

# ---------- high-level wrapper ----------
def run_ioc(model, data, tau, A_list, B_list, X, U, deltaT, initial_guess=None):
    n_state = X.shape[1]
    n_ctrl = U.shape[1]
    # initial params: small Q (identity), R = eye * 0.1
    # param vector = lower triang of L (for Q = L L^T) + log(R_diag)
    n_lt = n_state*(n_state+1)//2
    if initial_guess is None:
        L0 = np.eye(n_state) * 0.1
        L0_entries = []
        for i in range(n_state):
            for j in range(i+1):
                L0_entries.append(L0[i,j])
        logR0 = np.log(np.ones(n_ctrl)*0.1)
        params0 = np.hstack([L0_entries, logR0])
    else:
        params0 = initial_guess

    # run optimization
    res = minimize(lambda p: ioc_loss(p, A_list, B_list, X, U, n_state, n_ctrl),
                   params0, method='L-BFGS-B', options={'maxiter':300, 'disp':True})
    p_opt = res.x
    Q_opt, R_opt, Qf_opt = param_to_QR(p_opt, n_state, n_ctrl)
    return Q_opt, R_opt, Qf_opt, res

# Load dataset 
data_folder = "Data/S_393_DecreaseStrong/J1/Natural Before Exposition"
size =   160 #cm
weight = 70 #kg

arm_size = 0.186* size # 18.6% of height
forearm_size = 0.146* size # 14.6% of height 
arm_mass = 0.028* weight # 2.8% of weight
forearm_mass = 0.018* weight # 1.8% of weight
arm_com = 0.452* arm_size # 45.2% of arm length
forearm_com = 0.424* forearm_size # 42.4% of forearm length

g=[0,-9.81,0]

file_pattern = os.path.join(data_folder, "mocap_data_go*")
mocap_files = sorted(glob.glob(file_pattern))
mocap_dfs = [pd.read_csv(f) for f in mocap_files]
X = mocap_dfs

# create arm model
arm = Arm(arm_size/100, forearm_size/100) # lengths in meters
model = arm.create_DH_model( arm_mass, forearm_mass, arm_com/100, forearm_com/100) # masses in kg, com in meters
fig = model.plot(np.radians([0, 0, 0, 180-122,0]))
fig.ax.plot([0,0,0], [0,-0.6,0], [0,0,0], color='k', linewidth=2, label='Trunk')
plt.show(block=True)

n_sample = 23
# Play data
data = X[n_sample][['1', '2', '3', '4']]
data['5'] = 0
data['4'] = np.pi-data['4']
deltaT = X[n_sample]['time'].diff().mean()
# Convert joint angles from radians to degrees for plotting
data_deg = data.copy()
for col in ['1', '2', '3', '4']:
    data_deg[col] = np.degrees(data_deg[col])
data_deg.plot(title="Joint Angles (degrees)")
plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.show()
print(data)
# fig = model.plot(data.values)

plt.figure( )
plt.plot(data.diff()/deltaT)
plt.title("Joint Velocities (rad/s)")
plt.show()

tau = model.rne(data.values, (data.diff()/deltaT).values, (data.diff().diff()/(deltaT**2)).values, gravity=g)

plt.figure()
plt.plot(tau)
plt.title("Joint Torques (Nm)")
plt.show(block=True)

list_A = []
list_B = []

for i in np.arange(data.shape[0]):
    index = i
    A, B = arm.linearize_finite_diff(model,  data.values[index, :], (data.diff()/deltaT).values[index, :], tau[index, :], gravity=g, eps=1e-6)
    print("Linearized A:\n", A)
    print("Linearized B:\n", B)
    list_A.append(A)
    list_B.append(B)

X = np.hstack([data.values, (data.diff()/deltaT).fillna(0).values])
print(X.shape)

Q_opt, R_opt, Qf_opt, res = run_ioc(model, data, tau, list_A, list_B, X, tau, deltaT)

print("Recovered Q:\n", Q_opt)
print("Recovered R:\n", R_opt)
print("Recovered Qf:\n", Qf_opt)
print("Optimization result:\n", res)    