"""
Implémentation simple en Python du modèle Guigon et al. 2007 (2-DOF plan).
- Discrétisation directe des commandes (u constant par intervalle)
- Intégration RK4 de l'état:
    x = [q1, q2, v1, v2, a1,a2,a3,a4, e1,e2,e3,e4]
- Coût: sum(u**2)*dt + pénalités sur conditions terminales
Paramètres repris du papier (voir citations).
"""
import numpy as np
from scipy.optimize import minimize

# ---------- paramètres (valeurs du papier pour N=2) ----------
m1, L1, c1, I1 = 2.52, 0.33, 0.5, 0.023
m2, L2, c2, I2 = 1.3, 0.4, 0.5, 0.011
tau = 0.04            # constante temporelle muscle (s). Voir article. :contentReference[oaicite:4]{index=4}
gamma = 10.0          # paramètre de régularisation / phi (paper uses phi smoothing) (approx)
r_sh = 0.04           # moment arm equivalent shoulder, elbow (paper used 0.04) :contentReference[oaicite:5]{index=5}
r_el = 0.04

# ---------- utilitaires dynamiques ----------
def inertia_terms(q2):
    """Calcule A11, A12, A22, C1, C2 (formules du papier)."""
    Lc1 = c1 * L1
    Lc2 = c2 * L2
    A11 = I1 + I2 + m1*Lc1**2 + m2*(L1**2 + Lc2**2 + 2*L1*Lc2*np.cos(q2))
    A12 = I2 + m2*(Lc2**2 + L1*Lc2*np.cos(q2))
    A21 = A12
    A22 = I2 + m2*Lc2**2
    return A11, A12, A21, A22

def coriolis_terms(q2, v1, v2):
    Lc2 = c2 * L2
    C1 = -m2*L1*Lc2*(2*v1*v2 + v2**2)*np.sin(q2)
    C2 = m2*L1*Lc2*(v1**2)*np.sin(q2)
    return C1, C2

def phi(x):
    """approx différentiable de [z]_+ (pour la force -> force de traction seulement)."""
    # phi(z) ~ log(1+exp(k*z))/k
    k = 50.0
    return (np.log1p(np.exp(k*x)))/k

# ---------- modèle muscle (Eq.4) ----------
def muscle_dynamics(a, e, u):
    # tau * de/dt = -e + u
    # tau * da/dt = -a + e
    de = (-e + u) / tau
    da = (-a + e) / tau
    return da, de

# ---------- dynamique du système (angles, vitesses) ----------
def arm_accelerations(q1, q2, v1, v2, Tsh, Tel):
    # utilise les Aij et C terms
    A11, A12, A21, A22 = inertia_terms(q2)
    C1, C2 = coriolis_terms(q2, v1, v2)
    # On a: [A] * ddq + C = T  => ddq = A^{-1} (T - C)
    det = A11*A22 - A12*A21
    # si det petit -> num stable? on assume non singulier pour postures raisonnables
    ddq1 = ( (Tsh - C1)*A22 - (Tel - C2)*A12 ) / det
    ddq2 = ( (Tel - C2)*A11 - (Tsh - C1)*A21 ) / det
    return ddq1, ddq2

# ---------- intégrateur RK4 pour tout l'état ----------
def integrate_state(u_traj, q0, qf, tf, Nt):
    """
    u_traj: array (Nt, 4) -> 4 contrôles (u1..u4) (2 DOF => 4 muscles)
    q0: (q1_0,q2_0); qf: (q1_f,q2_f); tf: final time
    retourne t, states, controls
    """
    dt = tf / Nt
    # état initial
    q1, q2 = q0
    v1, v2 = 0.0, 0.0
    a = np.zeros(4)
    e = np.zeros(4)
    states = []
    t = 0.0
    for k in range(Nt):
        u = u_traj[k]  # shape (4,)
        # RK4 step for all variables
        def deriv(state, u):
            q1, q2, v1, v2, a0,a1,a2,a3, e0,e1,e2,e3 = state
            # torques from activations -> forces -> torques
            # shoulder torque Tsh = r_sh * ( F1 - F2 ), elbow Tel = r_el * (F3 - F4)
            F = np.array([phi(ai) for ai in (a0,a1,a2,a3)])  # forces positives seulement
            Tsh = r_sh * (F[0] - F[1])
            Tel = r_el * (F[2] - F[3])
            ddq1, ddq2 = arm_accelerations(q1,q2,v1,v2,Tsh,Tel)
            # muscle dynamics
            da0,de0 = muscle_dynamics(a0,e0,u[0])
            da1,de1 = muscle_dynamics(a1,e1,u[1])
            da2,de2 = muscle_dynamics(a2,e2,u[2])
            da3,de3 = muscle_dynamics(a3,e3,u[3])
            return np.array([v1, v2, ddq1, ddq2, da0,da1,da2,da3, de0,de1,de2,de3])
        # state vector
        S = np.array([q1,q2,v1,v2,a[0],a[1],a[2],a[3], e[0],e[1],e[2],e[3]])
        k1 = deriv(S, u)
        k2 = deriv(S + 0.5*dt*k1, u)
        k3 = deriv(S + 0.5*dt*k2, u)
        k4 = deriv(S + dt*k3, u)
        S = S + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        q1,q2,v1,v2 = S[0],S[1],S[2],S[3]
        a = np.array(S[4:8])
        e = np.array(S[8:12])
        t += dt
        states.append(np.concatenate(([t],[q1,q2,v1,v2],a,e)))
    times = np.linspace(dt, tf, Nt)
    return times, np.array(states)  # states shape (Nt, 1+...)


# ---------- fonction objectif pour l'optimisation ----------
def objective(flat_u, q0, qf, tf, Nt, w_pen=1e4):
    # flat_u length = Nt * 4
    u_traj = flat_u.reshape(Nt, 4)
    dt = tf / Nt
    # coût effort:
    effort = np.sum(u_traj**2) * dt
    # intégrer pour obtenir état final
    times, states = integrate_state(u_traj, q0, qf, tf, Nt)
    final = states[-1]
    # récupérer q1,q2,v1,v2
    q1f_sim, q2f_sim, v1f_sim, v2f_sim = final[1], final[2], final[3], final[4]
    # pénalité pour contrainte de bord (position & vitesse à tf)
    pen = (q1f_sim - qf[0])**2 + (q2f_sim - qf[1])**2 + v1f_sim**2 + v2f_sim**2
    return effort + w_pen * pen

# ---------- optimization wrapper ----------
def solve_2dof(q0, qf, tf=0.6, Nt=60):
    # initial guess: zeros
    init_u = np.zeros((Nt,4))
    flat0 = init_u.ravel()
    bounds = [(-5.0, 5.0)] * flat0.size  # bornes sur signaux de commande (ajuster si besoin)
    res = minimize(lambda z: objective(z, q0, qf, tf, Nt),
                   flat0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter':200, 'disp': True})
    u_opt = res.x.reshape(Nt,4)
    times, states = integrate_state(u_opt, q0, qf, tf, Nt)
    return times, states, u_opt, res

# ---------- exemple d'utilisation ----------
if __name__ == "__main__":
    q0 = (np.deg2rad(45), np.deg2rad(90))     # position initiale (radians)
    qf = (np.deg2rad(30), np.deg2rad(60))     # cible finale
    times, states, u_opt, res = solve_2dof(q0, qf, tf=0.6, Nt=80)
    print("optim status:", res.success, res.message)
