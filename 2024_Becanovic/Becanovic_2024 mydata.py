import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from Becanovic_2024_func import make_ndof_model, instantiate_ndof_model, numerize_var
from Becanovic_2024_plot import plot_snapshots_from_vars, plot_snapshots_from_vars, plot_joint_traj_from_vars, plot_segment_vels_from_vars
import os
import glob
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import os
import re


n = 2  # number of joints
n_segments = 2

list_theta = []

path_to_data = 'Data/S_376_DecreaseStrong/J1/Natural After Exposition/'
# extract last and penultimate segments between slashes
p_norm = os.path.normpath(path_to_data)
condition = os.path.basename(p_norm)

rmse_list = []


# Find all CSV files in path_to_data that start with 'mocap_data_go'
mocap_files = sorted(glob.glob(os.path.join(path_to_data, 'mocap_data_go*.csv')))



def forward_kinematics_2dof(theta1, theta2, L1, L2):
    """
    Compute the forward kinematics of a 2-DOF planar robot arm.

    Parameters:
    -----------
    theta1 : float
        Joint 1 angle in radians
    theta2 : float
        Joint 2 angle in radians
    L1 : float
        Length of link 1
    L2 : float
        Length of link 2

    Returns:
    --------
    x, y : float
        End-effector position in the plane
    """
    # Compute end-effector position
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    
    return x, y



if not mocap_files:
    print(f'No mocap files found in: {path_to_data}')
else:
    for fpath in mocap_files:
        print(fpath)
        try:
            data = pd.read_csv(fpath)
            print(data.columns)
           

        except Exception as e:
            print(f'Failed to load {fpath}: {e}')
            
        orig_N = len(data)
        # Quick plot of columns '1','2','3','4' vs time (if they exist)
        cols_to_plot = ['1', '2', '3', '4']
        existing_cols = [c for c in cols_to_plot if c in data.columns]


        
        # %% Prepare constants and variables for optimal control
        N = len(data)
        T = data['time'].iloc[-1] - data['time'].iloc[0]
        dt = T/(N-1)  # time step
        data['time'] = np.linspace(0, T, N)  # ensure time vector is consistent
        q0 = data.iloc[0][['3', '4']].to_numpy()  # initial position
        dq0 = np.zeros(n)  # initial velocity
        size =   1.80 #m
        weight = 76 #kg

        arm_size = 0.186 * size + 0.1# 18.6% of height
        forearm_size = 0.146 * size + 0.15 # 14.6% of height
        arm_mass = 0.028* weight # 2.8% of weight
        forearm_mass = 0.018* weight # 1.8% of weight
        arm_com = 0.452* arm_size # 45.2% of arm length
        forearm_com = 0.424* forearm_size # 42.4% of forearm length
        L = np.array([arm_size,  forearm_size])  # segment lengths
        COM = np.vstack((np.array([arm_com, forearm_com]).reshape(1,n_segments), np.zeros((1, n_segments))))
        M = np.array([arm_mass, forearm_mass])  # segment masses
        I = np.array([1/12 * arm_mass * arm_size**2, 1/12 * forearm_mass * forearm_size**2]) # segment inertias
        gravity = np.array([  0, -9.81])  # gravity vector
        Fext = [np.zeros((3, N - 2)) for _ in range(n)]
        goal = np.array(forward_kinematics_2dof(data['3'].iloc[-1], data['4'].iloc[-1], L[0], L[1]))
        
        
        # print("Goal position (m):", goal.flatten())
        # print("Start position (m):", np.array(forward_kinematics_2dof(q0[0], q0[1], L[0], L[1])))
        # print('L:', L)
        # print('COM:', COM)
        # print('M:', M)
        # print('I:', I)
        # print('gravity:', gravity)
        # print('q0:', q0)
  
        
        q = data[['3', '4']].to_numpy().T 
        dq = np.diff(q, axis=1) / dt
        ddq = np.diff(dq, axis=1) / dt 
        print(q)
        print(q.shape)
        
        #%% Plot input data 
        # Plot q, dq, ddq: n rows x 3 columns
        t = data['time'].to_numpy()
        t_q = t
        t_dq = t[1:]
        t_ddq = t[2:]
        
        q_verif = q
        dq_verif = dq
        ddq_verif = ddq
      
#%% Direct optimal control to find initialization of lambda and theta
        # Instanciate 
        opti, var = make_ndof_model(n, N) 
        instantiate_ndof_model(var, opti, dt, q0, dq0, L, COM, M, I, gravity, Fext, goal, ddq, dq, q);

        opti.solver('ipopt')

        # Optimize joint torque
        theta_1 = np.array([1,5000,1, 1, 1, 1, 1, 20, 1, 5, 1])  # weights for cost function 1
        theta_1 = theta_1 / np.linalg.norm(theta_1)

        opti.minimize(theta_1[0]* var['costs']['joint_vel_cost'] + theta_1[1]* var['costs']['joint_torque_cost'] + theta_1[2]* var['costs']['ee_vel_cost'] + theta_1[3]* var['costs']['joint_torque_change_cost'] + theta_1[4]* var['costs']['joint_jerk_cost'] + theta_1[5]* var['costs']['torque change_cost'] + theta_1[6]* var['costs']['acceleration_cost'] + theta_1[7]* var['costs']['mechanical_work_cost'] + theta_1[8]* var['costs']['duration_cost'] + theta_1[9]* var['costs']['accuracy_cost'] + theta_1[10]* var['costs']['posture_cost'])
        sol_1 = opti.solve()
        lambda_1 = sol_1.value(opti.lam_g) # Extract dual variables

 
        # Extract primal variables
        q_1 = sol_1.value(var['variables']['q'])
        dq_1 = sol_1.value(var['variables']['dq'])
        ddq_1 = sol_1.value(var['variables']['ddq'])
        
        # Numerize
        num_vars_1 = numerize_var(var, sol_1)
        
        q_exp = data[['3', '4']].to_numpy().reshape(N,n).T
        dq_exp = np.diff(q_exp, axis=1) / dt
        ddq_exp = np.diff(dq_exp, axis=1) / dt

        
#%% IOC identification
        ## Make IOC
        [opti_ioc, vars_ioc] = make_ndof_model(n, N);

        # Extract dual variables size
        ndual = len(lambda_1)
        # Extract parameters size
        nparam = len(theta_1)
        print(nparam)
    

        # Create dual variable parameter
        vars_ioc["variables"]["lambda"] = opti_ioc.variable(ndual)
        # Create model parameter
        vars_ioc["variables"]["theta"] = opti_ioc.variable(nparam)

        # Prepare stationarity constraint
        vars_ioc["costs"]["compound_cost"] = vars_ioc["variables"]["theta"][0] * vars_ioc["costs"]["joint_vel_cost"] + vars_ioc["variables"]["theta"][1] * vars_ioc["costs"]["joint_torque_cost"] + vars_ioc["variables"]["theta"][2] * vars_ioc["costs"]["ee_vel_cost"] + vars_ioc["variables"]["theta"][3] * vars_ioc["costs"]["joint_torque_change_cost"] + vars_ioc["variables"]["theta"][4] * vars_ioc["costs"]["joint_jerk_cost"] + vars_ioc["variables"]["theta"][5] * vars_ioc["costs"]["torque change_cost"] + vars_ioc["variables"]["theta"][6] * vars_ioc["costs"]["acceleration_cost"] + vars_ioc["variables"]["theta"][7] * vars_ioc["costs"]["mechanical_work_cost"] + vars_ioc["variables"]["theta"][8] * vars_ioc["costs"]["duration_cost"] + vars_ioc["variables"]["theta"][9] * vars_ioc["costs"]["accuracy_cost"] + vars_ioc["variables"]["theta"][10] * vars_ioc["costs"]["posture_cost"]

        q_vec = ca.vec(vars_ioc["variables"]["q"])
        dq_vec = ca.vec(vars_ioc["variables"]["dq"])
        ddq_vec = ca.vec(vars_ioc["variables"]["ddq"])

        all_vars = ca.vertcat(q_vec, dq_vec, ddq_vec)
       
        # -----------------------------
        # Compute gradient of compound cost w.r.t all variables
        # -----------------------------
        grad_compound_cost = ca.jacobian(vars_ioc["costs"]["compound_cost"], all_vars).T  # transpose to match MATLAB
        vars_ioc["costs"]["grad_compound_cost"] = grad_compound_cost
        


        init_pos = ca.vec(vars_ioc["constraints"]["initial_pos"]) 
        init_vel = ca.vec(vars_ioc["constraints"]["initial_vel"])
        dynamics_pos = ca.vec(vars_ioc["constraints"]["dynamics_pos"])
        dynamics_vel = ca.vec(vars_ioc["constraints"]["dynamics_vel"])
        goal_ee = ca.vec(vars_ioc["constraints"]["goal_ee"])

        # Concatenate all constraints
        compound_constraints = ca.vertcat(init_pos, init_vel, dynamics_pos, dynamics_vel, goal_ee)
        vars_ioc["constraints"]["compound_constraints"] = compound_constraints

        # Compute gradient of compound constraints w.r.t all variables (use the original all_vars)
        vars_ioc["constraints"]["grad_compound_constraints"] = ca.jacobian(compound_constraints, all_vars)

        vars_ioc["constraints"]["stationarity"] = vars_ioc["costs"]["grad_compound_cost"] + vars_ioc["constraints"]["grad_compound_constraints"].T @ vars_ioc["variables"]["lambda"]


        # 1. Stationarity constraint
        #opti_ioc.subject_to(vars_ioc["constraints"]["stationarity"] == 0)
        # Remove the hard equality constraint on stationarity
        stationarity_residual = vars_ioc["constraints"]["stationarity"]

        penalty_weight = 1e-3  # tune this
        penalty_KKT = penalty_weight * ca.sumsqr(stationarity_residual)

        # 2. Theta sum equals 1
        opti_ioc.subject_to(ca.sum1(vars_ioc["variables"]["theta"]) == 1)

        # 3. Theta non-negativity
        opti_ioc.subject_to(vars_ioc["variables"]["theta"] >= 0)

        # 4. Joint limits around q_1
        q = vars_ioc["variables"]["q"]

        # Flatten q - q_1
        q_diff = ca.vec(q - q_exp)
        opti_ioc.subject_to(q_diff <= np.pi / 2)
        opti_ioc.subject_to(-q_diff <= np.pi / 2)

        # -----------------------------
        # Create L2 loss
        # -----------------------------
        # MATLAB: vars_ioc.costs.L2_loss = sum(sum((vars_ioc.variables.q - q_1).^2));
        q_diff = vars_ioc["variables"]["q"] - q_exp
       
        vars_ioc["costs"]["L2_loss"] = ca.sumsqr(q_diff)  # sumsqr does exactly sum(sum(...^2))

        # -----------------------------
        # Minimize
        # -----------------------------
        opti_ioc.minimize(vars_ioc["costs"]["L2_loss"] + penalty_KKT)

        # -----------------------------
        # Instantiate model (set parameter values and initial guesses)
        # -----------------------------
        instantiate_ndof_model(
            var=vars_ioc,
            opti=opti_ioc,
            dt=dt,
            q0=q0,
            dq0=dq0,
            L=L,
            COM=COM,
            M=M,
            I=I,
            gravity=gravity,
            Fext=Fext,
            goal=goal,
            ddq=ddq_exp, 
            dq=dq_exp,
            q=q_exp
        )

        # -----------------------------
        # Set initial guesses for duals / theta
        # -----------------------------
 
        opti_ioc.set_initial(vars_ioc["variables"]["lambda"],lambda_1)
        opti_ioc.set_initial(vars_ioc["variables"]["theta"],  theta_1 )

        # -----------------------------
        # Solver options and solve
        # -----------------------------
        opts = {
            "ipopt.print_level": 5,           # see detailed progress
            "ipopt.max_iter": 5000,          # allow more iterations
            "ipopt.tol": 1e-3,                # loosen convergence tolerance
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 10,
            "ipopt.linear_solver": "mumps",   # robust linear solver
            "ipopt.mu_strategy": "adaptive",  # smoother barrier updates
            "ipopt.sb": "yes",                # suppress IPOPT banners
            'ipopt.obj_scaling_factor': 1e-3,  # Scale objective
            'ipopt.nlp_scaling_method': 'gradient-based',
            # 'ipopt.hessian_approximation': 'limited-memory'  # Use L-BFGS instead of exact Hessian

        }
 

        opti_ioc.solver('ipopt', opts)
        sol_ioc = opti_ioc.solve()

        # -----------------------------
        # Numerize
        # -----------------------------
        num_vars_ioc = numerize_var(vars_ioc, opti_ioc, initial_flag=False)
        
        print('Results : ', num_vars_ioc["costs"])
        list_theta.append((num_vars_ioc["costs"]['joint_torque_cost'], num_vars_ioc["costs"]['joint_vel_cost'], num_vars_ioc["costs"]['ee_vel_cost'], num_vars_ioc["costs"]['joint_torque_change_cost'], num_vars_ioc["costs"]['joint_jerk_cost'], num_vars_ioc["costs"]['torque change_cost'], num_vars_ioc["costs"]['acceleration_cost'], num_vars_ioc["costs"]['mechanical_work_cost'], num_vars_ioc["costs"]['duration_cost'], num_vars_ioc["costs"]['accuracy_cost'], num_vars_ioc["costs"]['posture_cost']))
     
        
       #%% Get reconstruced joint trajectories and compute RMSE
        q_ioc = num_vars_ioc["functions"]["q"]
        qd_ioc = num_vars_ioc["functions"]["dq"]
        qdd_ioc = num_vars_ioc["functions"]["ddq"]
        rmse = np.sqrt(np.mean((q_ioc - q_exp)**2, axis=1))  # RMSE per joint
        print("RMSE per joint (rad):", rmse)
        rmse_list.append(rmse)
        

        # -----------------------------
        # Snapshots plots
        # -----------------------------
        # plt.figure(figsize=(12, 8))

        # plot_snapshots_from_vars(num_vars_1, 10)
        # plt.plot(goal[0], goal[1], 'ro', markerfacecolor='auto', markersize=20)
        # plt.axis('equal')
        # plt.gca().xaxis.set_inverted(True)  # inverted axis with autoscaling
        # plt.gca().yaxis.set_inverted(True)  # inverted axis with autoscaling
        # plt.title("Snapshots: num_vars_1")


        # -----------------------------
        # Joint trajectories
        # -----------------------------
        plt.figure(figsize=(12, 8))            
        plt.title("Joint Trajectories" + fpath)

        plot_joint_traj_from_vars(num_vars_ioc)
        try:
            fig = plt.gcf()
            axes = fig.get_axes()
            nplots = len(axes)
            if nplots == 0:
                raise RuntimeError("No axes found to overlay data_exp.")

            # infer layout: assume n rows (one per joint), columns = total_axes // n
            ncols = max(1, nplots // n)

            # user requested "second and forth column"
            # use 0-based indices: 1 -> second column, 3 -> fourth column
            pos_col = 1
            vel_col = 2
            acc_col = 3

            # clamp to available columns (if requested column doesn't exist, use last column)
            vel_col = min(vel_col, ncols - 1)
            acc_col = min(acc_col, ncols - 1)

            for i in range(n):
                base = i * ncols

                # velocity subplot (second column or last if not available)
                vel_idx = base + vel_col
                if vel_idx < nplots:
                    ax_vel = axes[vel_idx]
                    # dq_exp has length N-1, use t_dq
                    ax_vel.plot(t_dq, dq_exp[i, :], color='C1', linewidth=1.2, linestyle='--', label='data_exp dq')
                    if i == 0:
                        ax_vel.legend()

                # acceleration subplot (fourth column if available, else last column)
                acc_idx = base + acc_col
                if acc_idx < nplots:
                    ax_acc = axes[acc_idx]
                    # ddq_exp has length N-2, use t_ddq
                    ax_acc.plot(t_ddq, ddq_exp[i, :], color='C2', linewidth=1.2, linestyle='--', label='data_exp ddq')
                    if i == 0:
                        ax_acc.legend()

                # also overlay q_exp on the primary position subplot if present (keeps previous behavior)
               
                pos_col = min(pos_col, ncols - 1)
                for i in range(n):
                    pos_idx = i * ncols + pos_col
                    if pos_idx < nplots:
                        ax_pos = axes[pos_idx]
                        ax_pos.plot(t_q, q_exp[i, :], color='C1', linewidth=1.5, linestyle='--', label='data_exp')
                        if i == 0:
                            ax_pos.legend()

        except Exception as e:
            print("Could not overlay data_exp:", e)
   

# Concatenate collected theta vectors and plot mean ± std (one bar per theta)
if len(list_theta) == 0:
    print("No theta results to plot.")
else:
    thetas = np.vstack(list_theta)  # shape: (n_runs, n_params)
    mean_theta = np.mean(thetas, axis=0)
    std_theta = np.std(thetas, axis=0)

    nparam = mean_theta.size
    x = np.arange(nparam)
    
    # Use cost names from the last vars_ioc if available, otherwise fallback to generic labels
    try:
        cost_keys = list(vars_ioc["costs"].keys())
        # filter out derived/internal cost entries
        filtered = [k for k in cost_keys if not any(s in k for s in ("compound", "grad", "L2"))]
        if len(filtered) >= nparam:
            labels = filtered[:nparam]
        else:
            labels = filtered + [f'theta_{i+1}' for i in range(len(filtered), nparam)]
    except Exception:
        labels = [f'theta_{i+1}' for i in range(nparam)]

    plt.figure(figsize=(8, 4))
    plt.bar(x, mean_theta, yerr=std_theta, capsize=6, color='C0', edgecolor='k')
    plt.xticks(x, labels)
    plt.ylabel('Weight (mean ± std)')
    plt.title('IOC weights across recordings')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    
    plt.figure(figsize=(8, 4))
    arr = np.array(rmse_list)
    if arr.ndim == 1:
        plt.plot(arr, 'o-', color='C0', markersize=8)
    else:
        x = np.arange(arr.shape[0])
        ncols = arr.shape[1]
        for i in range(ncols):
            plt.plot(x, arr[:, i], 'o-', color=f'C{i%10}', markersize=8, label=f'joint_{i+1}')
        plt.legend()
    plt.ylabel('RMSE (rad)')
    plt.title('RMSE of reconstructed joint trajectories')
  
    
    outdir = os.path.join(os.getcwd(), "figures_pdf/"+condition)
    os.makedirs(outdir, exist_ok=True)

    fignums = plt.get_fignums()
    if not fignums:
        print("No open figures to save.")
    else:
        for idx, num in enumerate(fignums, start=1):
            fig = plt.figure(num)
            # try to find a descriptive title
            title = ""
            if getattr(fig, "_suptitle", None):
                title = fig._suptitle.get_text() or ""
            if not title:
                for ax in fig.axes:
                    t = ax.get_title()
                    if t:
                        title = t
                        break
            if not title:
                title = f"figure_{idx}"
            # sanitize filename
            fname_base = re.sub(r"[^0-9A-Za-z._-]+", "_", title).strip("_")[:80]
            fname = f"{idx:02d}_{fname_base}.pdf"
            fpath = os.path.join(outdir, fname)
            fig.savefig(fpath, format="pdf", bbox_inches="tight")
            print(f"Saved figure #{num} -> {fpath}")
        plt.show()
        # optionally close all figures to free memory
        plt.close("all")
  

