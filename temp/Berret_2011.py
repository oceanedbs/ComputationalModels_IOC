import numpy as np
import casadi as ca
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# ---------- PARAMETERS (typical anthropometrics; replace with subject-specific) ----------
m1, m2 = 1.8, 1.2        # segment masses (kg) - set plausible values
l1, l2 = 0.34, 0.34      # segment lengths (m)
lc1, lc2 = l1*0.44, l2*0.44  # centers of mass (m)
I1, I2 = 0.014, 0.018    # moments of inertia (kg*m^2)
g = 9.81
visc = 0.01              # viscous term F (scalar multiply identity) - small damping
# actuator dynamics parameter (torque second derivative controlled)
# In paper: actuator: ddot(t) = m (control) -> we model second derivative of torque equals motor command
# we'll set actuator time scaling if necessary

# ---------- HELPER: robot dynamics M(h), C(h,hdot), G(h) ----------
def robot_dynamics_symbols():
    h1, h2 = ca.SX.sym('h1'), ca.SX.sym('h2')
    dh1, dh2 = ca.SX.sym('dh1'), ca.SX.sym('dh2')

    # shorthand
    c2 = ca.cos(h2)
    s2 = ca.sin(h2)

    # inertia matrix M(h)
    M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
    M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
    M21 = M12
    M22 = I2 + m2*lc2**2
    M = ca.vertcat(
        ca.horzcat(M11, M12),
        ca.horzcat(M21, M22)
    )

    # Coriolis/Centrifugal (approximation using Christoffel-like terms)
    h = ca.vertcat(h1, h2)
    dh = ca.vertcat(dh1, dh2)
    # compute C(h,dh)*dh term (standard 2-link analytic form)
    h2dot = dh2
    h1dot = dh1
    # elements (using standard robot formulas)
    h = h2 # reuse var (not used)
    H = -m2*l1*lc2*s2
    C1 = H*h2dot*(2*h1dot + h2dot)
    C2 = H*h1dot*h1dot
    C = ca.vertcat(C1, C2)

    # gravity vector G(h)
    G1 = (m1*lc1 + m2*l1)*g*ca.cos(h1) + m2*lc2*g*ca.cos(h1+h2)
    G2 = m2*lc2*g*ca.cos(h1+h2)
    G_vec = ca.vertcat(G1, G2)

    return {'h': ca.vertcat(h1,h2), 'dh': ca.vertcat(dh1,dh2),
            'M': M, 'C': C, 'G': G_vec}

# build CasADi functions
sym = robot_dynamics_symbols()
M_fun = ca.Function('M_fun', [sym['h']], [sym['M']])
C_fun = ca.Function('C_fun', [sym['h'], sym['dh']], [sym['C']])
G_fun = ca.Function('G_fun', [sym['h']], [sym['G']])

# ---------- DEFINE COSTS C1..C8 (continuous-time integrands) ----------
# We'll define them as functions that take state/control time series or CasADi symbolic variables.
# State q = [h1,h2, dh1, dh2, t1, t2, dt1, dt2] where t are torques, dt are torque derivatives.
# But paper states control m is torque acceleration: ddot(t) = m.

def integrand_costs(q_sym, u_sym):
    # q_sym is dictionary of CasADi SX symbols for states at a given time
    h = q_sym['h']          # 2x1
    dh = q_sym['dh']        # 2x1
    t = q_sym['t']          # 2x1 torques
    dt = q_sym['dt']        # 2x1 torque derivatives (first derivative)
    m = u_sym               # 2x1 motor commands (torque accelerations)

    # hand (Cartesian) jerk -> compute x_dddot^2 + y_dddot^2 requires forward kinematics derivatives.
    # For simplicity, we implement C1 as zero here OR approximate via joint jerk projection.
    # We'll implement main costs that paper emphasizes: angle acceleration (C3) and energy (C7).
    C = {}
    # C2: angle jerk (h''' squared) -- we would need third derivative of angles; in collocation we can get derivatives
    # Simpler: include C3 (integrated squared joint accelerations) using ddh (joint accelerations) available via dynamics
    # But our state will not explicitly include ddh; direct collocation enforces dynamics: M ddh + C + G + visc*dh = t
    # So ddh computed as solve(M, t - C - G - visc*dh)
    # We'll leave generic placeholders and compute most important costs below.

    # placeholder: return dictionary with lambdas outside (we compute inside collocation loop)
    return C

# ---------- SETUP DIRECT OCP (single shooting with collocation nodes) ----------
def solve_direct_ocp(weights, tf=0.75, N=60, init_state=None, h_init=None, h_final_constraint=None):
    # weights: vector of 8 non-negative weights [a1..a8] matching C1..C8 in Table 1
    # tf: movement duration
    # N: number of control intervals (collocation via direct transcription)
    nx = 8  # state: h1,h2,dh1,dh2, t1,t2, dt1,dt2  (torque and its derivative)
    nu = 2  # control: motor command m (ddot torque)
    opti = ca.Opti()

    # time grid
    dt = tf / N
    # decision variables
    X = opti.variable(nx, N+1)
    U = opti.variable(nu, N)  # motor commands (ddot torque)
    T = tf

    # unpack helpers
    def unpack_x(xcol):
        h = xcol[0:2]
        dh = xcol[2:4]
        t = xcol[4:6]
        dtq = xcol[6:8]
        return h, dh, t, dtq

    # initial conditions (use provided or default)
    if init_state is None:
        # default initial: both joints about horizontal -> example
        h0 = np.array([np.deg2rad(90), np.deg2rad(-90)])  # example (replace per posture)
        dh0 = np.zeros(2)
        t0 = np.zeros(2)
        dt0 = np.zeros(2)
    else:
        h0, dh0, t0, dt0 = init_state

    # dynamic constraints: use implicit integration (Euler-explicit for brevity)
    cost = 0
    for k in range(N):
        xk = X[:,k]
        xk1 = X[:,k+1]
        uk = U[:,k]

        h_k, dh_k, t_k, dt_k = unpack_x(xk)
        h_k1, dh_k1, t_k1, dt_k1 = unpack_x(xk1)

        # dynamics:
        # torque accel: ddot(t) = m  => dt_dot = uk
        # integrate dt: dt_{k+1} = dt_k + dt*uk
        opti.subject_to(dt_k1 == dt_k + dt*uk)

        # torque derivative: t_dot = dt_k  => t_{k+1} = t_k + dt*dt_k
        opti.subject_to(t_k1 == t_k + dt*dt_k)

        # joint acceleration from dynamics: M(h) ddh = t - C(h,dh) - G(h) - F*dh
        M_mat = M_fun(h_k)
        Cvec = C_fun(h_k, dh_k)
        Gvec = G_fun(h_k)
        # ddh = M^{-1} * (t - C - G - F*dh)
        ddh_k = ca.solve(M_mat, (t_k - Cvec - Gvec - visc*dh_k))
        # integrate dh: dh_{k+1} = dh_k + dt*ddh_k
        opti.subject_to(dh_k1 == dh_k + dt*ddh_k)
        # integrate h: h_{k+1} = h_k + dt*dh_k
        opti.subject_to(h_k1 == h_k + dt*dh_k)

        # compute instantaneous cost contributions (discretized trapezoid simple: f(xk))
        # Build the 8 cost integrands per Table 1. We'll explicitly implement the main ones:
        # C2 (angle jerk) not implemented here; C3 (angle acceleration) -> ddh_k^2
        c3 = ca.dot(ddh_k, ddh_k)          # integrated squared joint accelerations
        # C7 energy = sum | dh * t | -> mechanical power abs( hdot * torque )
        power = ca.fabs(dh_k[0]*t_k[0]) + ca.fabs(dh_k[1]*t_k[1])
        c7 = power
        # Simple approximate other costs as small terms (to make all 8 implemented):
        # C1 (hand jerk) omitted in detailed form (would require kinematic derivatives)
        c1 = 0
        c2 = 0
        c4 = ca.dot(dt_k, dt_k)   # squared torque derivative
        c5 = ca.dot(t_k, t_k)  # torque squared
        c6 = ca.norm_2(dh_k)   # geodesic-like simple proxy (not exact)
        c8 = ca.dot(uk, uk)    # motor command squared (effort)

        costs = [c1, c2, c3, c4, c5, c6, c7, c8]
        # accumulate weighted sum
        w = weights
        # simple Riemann sum:
        cost += dt * sum([w[i]*costs[i] for i in range(8)])

    # boundary conditions: initial state
    opti.subject_to(X[0:2,0] == h0)    # h init
    opti.subject_to(X[2:4,0] == dh0)   # dh init
    opti.subject_to(X[4:6,0] == t0)    # torque init
    opti.subject_to(X[6:8,0] == dt0)   # torque derivative init

    # terminal conditions: reach bar constraint => map joint angles to x,y and constrain x to bar
    # forward kinematics x = l1*cos(h1) + l2*cos(h1+h2) ; in paper x=0.85*L (bar location)
    L = l1 + l2
    x_final = l1*ca.cos(X[0, -1]) + l2*ca.cos(X[0, -1] + X[1, -1])
    y_final = l1*ca.sin(X[0, -1]) + l2*ca.sin(X[0, -1] + X[1, -1])
    # enforce x coordinate equals bar (example 0.85*L), zero velocity at end
    xb = 0.85 * L
    opti.subject_to(x_final == xb)
    opti.subject_to(X[2:4, -1] == ca.DM([0,0]))  # zero joint velocities at end
    # Also zero torque derivatives and velocities at end (paper)
    opti.subject_to(X[6:8, -1] == ca.DM([0,0]))
    opti.subject_to(X[4:6, -1] == ca.DM([0,0]))

    # bounds / simple regularization
    opti.subject_to(opti.bounded(-5, U, 5))  # limit motor command magnitude
    opti.minimize(cost)

    # initial guesses for solver
    opti.set_initial(X, 0)
    opti.set_initial(U, 0)

    # solver options
    p_opts = {"verbose": False}
    s_opts = {"max_iter": 2000}
    opti.solver('ipopt', p_opts, s_opts)
    try:
        sol = opti.solve()
    except Exception as e:
        print("Solver failed:", e)
        return None

    Xsol = sol.value(X)
    Usol = sol.value(U)
    time = np.linspace(0, tf, N+1)
    return {'time': time, 'X': Xsol, 'U': Usol}

# ---------- Example usage: simulate hybrid model C = 0.1*C3 + 1*C7 ----------
if __name__ == '__main__':
    # weight vector a1..a8
    weights = np.zeros(8)
    weights[2] = 0.1  # C3 angle acceleration
    weights[6] = 1.0  # C7 energy (absolute work) -- index 6 because python 0-based for C1..C8

    sol = solve_direct_ocp(weights, tf=0.75, N=80)
    if sol is not None:
        T = sol['time']
        X = sol['X']
        # plot fingertip path
        h1 = X[0,:]
        h2 = X[1,:]
        x = l1*np.cos(h1) + l2*np.cos(h1+h2)
        y = l1*np.sin(h1) + l2*np.sin(h1+h2)
        plt.figure()
        plt.plot(x, y, '-o', markevery=8)
        plt.axvline(0.85*(l1+l2), color='k', linestyle='--', label='bar')
        plt.gca().set_aspect('equal', 'box')
        plt.title('Simulated fingertip path (hybrid)')
        plt.legend()
        plt.show()
    else:
        print("No solution obtained.")
