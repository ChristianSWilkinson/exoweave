import os
import sys

# =============================================================================
# 0. THE SLURM THREAD LEASH (MUST BE BEFORE NUMPY/PANDAS IMPORTS)
# =============================================================================
# Strictly limit all C-level math libraries to 1 thread per Python process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import concurrent.futures
import pandas as pd
import numpy as np

# Import ExoWeave
from exoweave.coupler import ExoCoupler
from fuzzycore import constants as c
from fuzzycore.constants import G_CONST, M_EARTH, R_EARTH, M_JUPITER

# =============================================================================
# 1. DEFINE YOUR PARAMETER ARRAYS
# =============================================================================
T_INT_TARGETS = [200.0, 350.0, 500.0]
M_WATER_TARGETS = [0.0, 2.0, 4.0]
SIGMA_TARGETS = [0.001, 0.1, 0.3, 0.6]

# Planet Mass (GJ 1214 b)
TARGET_MASS_MJUP = 8.17 * (c.M_EARTH / c.M_JUPITER)

# Shared solver configuration
GLOBAL_CONFIG = {
    "max_iterations": 15,              
    "mass_convergence_threshold": 0.01,
    "p_bottom_bar": 1000.0,
    "resolution": 50,           
    "target_resolution": 50,
    "min_p_link_bar": 1,
    "retrieval_flux_error_bottom": 1e-4,
    "retrieval_flux_error_top": 1e-4,
    "n_iterations": 101,                      
    "n_non_adiabatic_iterations": 0,
    "chemistry_iteration_interval": 5,
    "cloud_iteration_interval": 0,
    "n_burn_iterations": 0,
    "retrieval_tolerance": 0.0,
    "weight_apriori": 0.001  
}

# =============================================================================
# 2. THE ISOLATED WORKER FUNCTION
# =============================================================================
def run_coupled_model(task_kwargs):
    """
    Standalone worker function that handles a single point on the grid.
    Takes a dictionary of parameters to ensure easy unpacking.
    """
    t_int = task_kwargs['t_int']
    m_water = task_kwargs['m_water']
    sigma = task_kwargs['sigma']
    m_core = task_kwargs['m_core']
    
    # Create a completely unique output directory for this specific run
    # This prevents Fortran file I/O collisions!
    run_dir = f"./outputs/subneptunes/Tint_{int(t_int)}/w_{m_water:.1f}_s_{sigma:.3f}"
    
    run_params = {
        "mass": TARGET_MASS_MJUP,
        "T_irr": 550.0,
        "T_int": t_int,
        "Met": 1.0,
        "core_mass_earth": m_core,
        "M_water": m_water,
        "sigma_val": sigma,
        "iron_fraction": 0.33,
        "f_sed": 3.0,
        "kzz": 8.0,
        "g_1bar": task_kwargs['g_init'],
        "initial_log_pc": 6.0,
        "debug": False
    }

    run_config = GLOBAL_CONFIG.copy()
    run_config["output_dir"] = run_dir

    try:
        coupler = ExoCoupler(target_params=run_params, config=run_config)
        results = coupler.run()
        
        status = results.get('status', 'failed')
        iters = results.get('iterations', 'N/A')
        return f"✅ SUCCESS: T_int={t_int}, W={m_water:.1f}, Sig={sigma:.3f} | Iters: {iters}" if status == 'converged' else f"⚠️ FAILED (Max Iters): T_int={t_int}, W={m_water:.1f}, Sig={sigma:.3f}"
        
    except Exception as e:
        return f"❌ CRASH: T_int={t_int}, W={m_water:.1f}, Sig={sigma:.3f} | Error: {e}"

# =============================================================================
# 3. MAIN EXECUTION BLOCK (Required for Multiprocessing)
# =============================================================================
if __name__ == "__main__":
    # Dynamically grab the number of CPUs allocated by SLURM. Fallback to 4.
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 24))
    print(f"🚀 Initializing Grid Run across {num_cpus} CPU cores...")

    print("📚 Loading waterworld_atlas.csv to validate M_core parameters...")
    try:
        atlas_df = pd.read_csv("../../fuzzycore/data/waterworld_atlas.csv")
        ok_df = atlas_df[atlas_df['Status'] == 'ok']
    except FileNotFoundError:
        print("❌ Error: waterworld_atlas.csv not found in the working directory.")
        sys.exit(1)

    # Target radius and tolerance for atlas-point filtering
    R_TARGET_RE = 2.75
    R_TOLERANCE = 1.00  # widen if too few points survive

    # Filter the atlas to converged points near the target radius FIRST
    viable = ok_df[np.abs(ok_df['R_total_Re'] - R_TARGET_RE) < R_TOLERANCE].copy()
    if len(viable) == 0:
        raise ValueError("No atlas points within R tolerance — widen R_TOLERANCE.")

    print(f"📍 Atlas filtered to {len(viable)} viable points near R={R_TARGET_RE} R_E")

    # Build the task list
    tasks = []
    max_w = viable['M_water_Me'].max() or 1.0
    max_s = viable['Sigma'].max() or 1.0

    for t, w, s in itertools.product(T_INT_TARGETS, M_WATER_TARGETS, SIGMA_TARGETS):
        # Normalize the differences to find the closest physically valid match in the atlas
        w_diff = np.abs(ok_df['M_water_Me'] - w) / max_w
        s_diff = np.abs(ok_df['Sigma'] - s) / max_s
        
        closest_idx = (w_diff + s_diff).idxmin()
        best_match = viable.loc[closest_idx]

        atlas_R_m  = float(best_match['R_total_Re']) * R_EARTH
        target_M_kg = TARGET_MASS_MJUP * M_JUPITER
        g_init = G_CONST * target_M_kg / atlas_R_m**2
        
        tasks.append({
            't_int': t,
            'm_water': float(best_match['M_water_Me']),
            'sigma': float(best_match['Sigma']),
            'm_core': float(best_match['M_core_Me'])
        })

    total_tasks = len(tasks)
    print(f"📦 Assembled {total_tasks} unique grid configurations.")
    print(f"⚙️ Dispatching to ProcessPoolExecutor...")

    # Execute in parallel
    results_log = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_coupled_model, task): task for task in tasks}
        
        # Process results as they finish (out of order, which is fine)
        for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
            try:
                result_msg = future.result()
                print(f"[{i+1}/{total_tasks}] {result_msg}", flush=True)
                results_log.append(result_msg)
            except Exception as exc:
                print(f"[{i+1}/{total_tasks}] 💥 Uncaught Exception in worker: {exc}", flush=True)

    print("\n🎉 All grid tasks completed.")