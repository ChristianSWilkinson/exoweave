import os
import time
import logging
import itertools
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import ExoWeave Coupler
from exoweave.coupler import ExoCoupler

# Suppress debug noise from individual processes
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# 1. GRID DEFINITION
# =============================================================================
# Define the parameter axes you want to sweep across. 

MASSES_MJUP   = [0.5, 1.0, 2.0]          # Jupiter Masses
T_INTS_K      = [200.0, 400.0, 600.0]    # Internal Temperatures
T_IRRS_K      = [1000.0]                 # Irradiation
METS          = [0.0, 0.5]               # Metallicity (log10 Z/Z_solar)
CORE_MASSES_E = [10.0, 15.0]             # Solid core mass in Earth masses
F_SEDS        = [1.0, 3.0]               # Cloud sedimentation
KZZS          = [8.0, 9.0]               # Eddy diffusion (log10)
# -------------------------------------

# Static parameters shared across all models in the grid
STATIC_PARAMS = {
    "iron_fraction": 0.33,
    "debug": False  
}

# Global configuration for the Coupler
GRID_CONFIG = {
    "output_dir": "outputs/grid_run",  
    "max_iterations": 15,
    "mass_convergence_threshold": 0.01,
    "p_bottom_bar": 1000.0,
    
    # --- Resolution Setup ---
    "resolution": 50,           # The fast, iterative solving resolution
    "target_resolution": 500    # The final, 1-time forward pass resolution!
}

# =============================================================================
# 2. WORKER FUNCTION
# =============================================================================
def run_model(target_params: dict) -> dict:
    try:
        import random
        time.sleep(random.uniform(0.1, 1.0))
        
        coupler = ExoCoupler(target_params=target_params, config=GRID_CONFIG)
        results = coupler.run()
        
        return {
            "mass": target_params["mass"],
            "T_int": target_params["T_int"],
            "core": target_params["core_mass_earth"],
            "status": results["status"],
            "iterations": results.get("iterations", "N/A"),
        }
    except Exception as e:
        return {
            "mass": target_params["mass"],
            "T_int": target_params["T_int"],
            "core": target_params["core_mass_earth"],
            "status": f"crashed: {str(e)}",
            "iterations": "N/A" 
        }

# =============================================================================
# 3. MAIN EXECUTION POOL
# =============================================================================
if __name__ == "__main__":
    # Add the new parameters to the combination axes
    axes = [MASSES_MJUP, T_INTS_K, T_IRRS_K, METS, CORE_MASSES_E, F_SEDS, KZZS]
    combinations = list(itertools.product(*axes))
    
    # --- NEW: PRE-FLIGHT CACHE SCANNER ---
    output_dir = Path(GRID_CONFIG["output_dir"])
    completed_tasks = set()
    
    if output_dir.exists():
        print(f"🔍 Scanning {output_dir} for previously completed models...")
        for pkl_file in output_dir.glob("**/*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # We only want to skip files that successfully finished (or maxed out)
                # We do NOT skip 'failed' or 'crashed' models so the grid can try them again!
                status = data.get('status', '')
                if status in ['converged', 'max_iterations_reached']:
                    
                    p = data.get('final_params', data.get('parameters', {}))
                    
                    # CRITICAL: We must use T_int_input_dial! 
                    # The true T_int shifts during convergence, but we need to match the original grid axis
                    target_tint = p.get('T_int_input_dial', p.get('T_int'))
                    
                    # Build a unique tuple of the inputs
                    task_tuple = (
                        p.get('mass'),
                        target_tint,
                        p.get('T_irr'),
                        p.get('Met'),
                        p.get('core_mass_earth'),
                        p.get('f_sed'),
                        p.get('kzz')
                    )
                    
                    # Safely round them to avoid floating-point mismatches during set comparison
                    if all(v is not None for v in task_tuple):
                        rounded_tuple = tuple(round(float(v), 4) for v in task_tuple)
                        completed_tasks.add(rounded_tuple)
            except Exception:
                continue
                
        print(f"✅ Found {len(completed_tasks)} already completed models. They will be skipped.")
        print("-" * 60)
        
    # Build the specific dictionary for each grid point
    grid_tasks = []
    
    # Unpack the 7 variables now!
    for (m, tint, tirr, met, core, fsed, kzz) in combinations:
        
        # --- NEW: Check if this combination is already in our completed set ---
        current_tuple = (m, tint, tirr, met, core, fsed, kzz)
        rounded_current = tuple(round(float(v), 4) for v in current_tuple)
        
        if rounded_current in completed_tasks:
            continue  # Skip this point, it's already safely on disk!
            
        params = STATIC_PARAMS.copy()
        params.update({
            "mass": m,
            "T_int": tint,
            "T_irr": tirr,
            "Met": met,
            "core_mass_earth": core,
            "f_sed": fsed,
            "kzz": kzz
        })
        grid_tasks.append(params)
        
    total_models = len(grid_tasks)
    
    if total_models == 0:
        print("🎉 All grid combinations have already been successfully computed! Nothing to do.")
        exit(0)
        
    print(f"🚀 INITIALIZING EXOWEAVE GRID COMPUTING...")
    print(f"📦 Remaining Models to Compute: {total_models}")
    print(f"💻 CPU Cores Detected: {os.cpu_count()}")
    print("-" * 60)

    start_time = time.time()
    
    max_cores = max(1, os.cpu_count() - 2) 
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        future_to_params = {executor.submit(run_model, p): p for p in grid_tasks}
        
        for i, future in enumerate(as_completed(future_to_params), 1):
            res = future.result()
            
            if res["status"] == "converged":
                successful += 1
                icon = "✅"
            else:
                failed += 1
                icon = "❌"
                
            print(f"[{i}/{total_models}] {icon} M={res['mass']} | "
                  f"T_int={res['T_int']} | Core={res['core']} | Status: {res['status']} "
                  f"(Iters: {res['iterations']})")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"🏁 GRID COMPLETE IN {elapsed/60:.2f} MINUTES.")
    print(f"📊 Success Rate for this batch: {successful}/{total_models} ({(successful/total_models)*100:.1f}%)")
    print(f"💾 All converged models are safely stored in: {GRID_CONFIG['output_dir']}")