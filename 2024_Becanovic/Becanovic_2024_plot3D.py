import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_snapshots_from_vars(fig, var, nsnapshots=10):
    """
    Plots 3D snapshots of the robot in a single figure.

    Args:
        var (dict): Dictionary containing variables and functions (same as in the model)
        nsnapshots (int): Number of snapshots to display along the trajectory
    """
    q = var["variables"]["q"]
    n, N = q.shape  # number of joints, timesteps

    ax = fig.add_subplot(111, projection='3d')

    # -----------------------------
    # End-effector trajectory
    # -----------------------------
    P_end = var["functions"]["P"][-1]  # 3×N array
    ax.plot(P_end[0, :], P_end[1, :], P_end[2, :],
            color='y', linewidth=2, label="EE traj")

    # -----------------------------
    # Segment COM trajectories
    # -----------------------------
    for ii in range(n):
        Pcom = var["functions"]["Pcom"][ii]
        ax.plot(Pcom[0, :], Pcom[1, :], Pcom[2, :],
                color='c', linewidth=2, alpha=0.2)

    # -----------------------------
    # Total COM trajectory (if exists)
    # -----------------------------
    if "Pcomtotal" in var["functions"]:
        Pcomtotal = var["functions"]["Pcomtotal"]
        ax.plot(Pcomtotal[0, :], Pcomtotal[1, :], Pcomtotal[2, :],
                color=[0.2, 0.7, 0.05], linewidth=2, label="Total COM")

    # -----------------------------
    # Plot snapshots of the robot
    # -----------------------------
    snapshot_indices = np.floor(np.linspace(0, N-1, nsnapshots)).astype(int)

    for ii in snapshot_indices:
        P, _, _, _ = snapshot_from_vars(var, ii, ax=ax)
        alpha = 0.2
        dP = P[-1, :] - P[-2, :]
        ax.text(P[-1, 0] + alpha * dP[0],
                P[-1, 1] + alpha * dP[1],
                P[-1, 2] + alpha * dP[2],
                f"{ii}", fontsize=12, color='k')

    # Styling
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # equal scale
    plt.tight_layout()


def snapshot_from_vars(var, ii, ax=None, marker_size=6, line_width=2):
    """
    Draws a 3D snapshot of the robot at timestep ii.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    q = var["variables"]["q"]
    n = q.shape[0]

    # -----------------------------
    # Link positions
    # -----------------------------
    P = np.vstack([np.array(Pi) for Pi in var["functions"]["P"]])  # shape (3*(n+1), N)
    Px = P[0::3, ii]
    Py = P[1::3, ii]
    Pz = P[2::3, ii]

    # -----------------------------
    # COM positions
    # -----------------------------
    Pcom = np.hstack(var["functions"]["Pcom"])
    Pcomx = Pcom[0::3, ii]
    Pcomy = Pcom[1::3, ii]
    Pcomz = Pcom[2::3, ii]

    # Optional total COM
    has_Pcomtotal = "Pcomtotal" in var["functions"]
    if has_Pcomtotal:
        Pcomtotalx = var["functions"]["Pcomtotal"][0, ii]
        Pcomtotaly = var["functions"]["Pcomtotal"][1, ii]
        Pcomtotalz = var["functions"]["Pcomtotal"][2, ii]

    # -----------------------------
    # Plot robot skeleton
    # -----------------------------
    ax.plot(Px, Py, Pz, color='k', marker='o', linestyle='-',
            markersize=marker_size, linewidth=line_width)

    # -----------------------------
    # Plot COM markers
    # -----------------------------
    ax.scatter(Pcomx, Pcomy, Pcomz, color='c', s=50, label='Link COMs')

    if has_Pcomtotal:
        ax.scatter(Pcomtotalx, Pcomtotaly, Pcomtotalz, color=[0.2, 0.7, 0.05],
                   marker='s', s=70, label='Total COM')

    return np.column_stack((Px, Py, Pz)), None, None, np.column_stack((Pcomx, Pcomy, Pcomz))

