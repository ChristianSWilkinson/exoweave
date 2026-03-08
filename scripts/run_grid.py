import os
import time
import logging
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import ExoWeave Coupler
from exoweave.coupler import ExoCoupler

# Suppress debug noise from individual processes
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# 1. GRID DEFINITION (RANDOMIZED)
# =============================================================================
# Define the Min and Max bounds for each parameter axis.
# If you want a parameter to remain static (like T_irr), just set min and max to the same value.

PARAM_BOUNDS = {
    "mass": (0.1, 13.0),             # Jupiter Masses
    "T_int": (200.0, 1500.0),        # Internal Temperatures (K)
    "T_irr": (100.0, 1500.0),       # Irradiation (K) - Static example
    "Met": (-2, 2),                 # Metallicity (log10 Z/Z_solar)
    "core_mass_earth": (1, 100),    # Solid core mass in Earth masses
    "f_sed": (1.0, 7.0),            # Cloud sedimentation
    "kzz": (0.0, 12.0)               # Eddy diffusion (log10)
}

TOTAL_RANDOM_MODELS = 1000  # Define how many total models you want to generate
RANDOM_SEED = 42           # Crucial for resumability! Do not change once a run starts.
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
    "resolution": 50,           
    "target_resolution": 500    
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
    
    print(f"🎲 Generating {TOTAL_RANDOM_MODELS} random grid points (Seed: {RANDOM_SEED})...")
    np.random.seed(RANDOM_SEED)
    
    combinations = []
    for _ in range(TOTAL_RANDOM_MODELS):
        # Draw uniform random samples across all defined bounds
        m = np.random.uniform(*PARAM_BOUNDS["mass"])
        tint = np.random.uniform(*PARAM_BOUNDS["T_int"])
        tirr = np.random.uniform(*PARAM_BOUNDS["T_irr"])
        met = np.random.uniform(*PARAM_BOUNDS["Met"])
        core = np.random.uniform(*PARAM_BOUNDS["core_mass_earth"])
        fsed = np.random.uniform(*PARAM_BOUNDS["f_sed"])
        kzz = np.random.uniform(*PARAM_BOUNDS["kzz"])
        
        combinations.append((m, tint, tirr, met, core, fsed, kzz))
    
    # --- PRE-FLIGHT CACHE SCANNER ---
    output_dir = Path(GRID_CONFIG["output_dir"])
    completed_tasks = set()
    
    if output_dir.exists():
        print(f"🔍 Scanning {output_dir} for previously completed models...")
        for pkl_file in output_dir.glob("**/*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    
                status = data.get('status', '')
                if status in ['converged', 'max_iterations_reached']:
                    
                    p = data.get('final_params', data.get('parameters', {}))
                    target_tint = p.get('T_int_input_dial', p.get('T_int'))
                    
                    task_tuple = (
                        p.get('mass'),
                        target_tint,
                        p.get('T_irr'),
                        p.get('Met'),
                        p.get('core_mass_earth'),
                        p.get('f_sed'),
                        p.get('kzz')
                    )
                    
                    if all(v is not None for v in task_tuple):
                        # Rounding to 4 decimals prevents random float precision errors
                        rounded_tuple = tuple(round(float(v), 4) for v in task_tuple)
                        completed_tasks.add(rounded_tuple)
            except Exception:
                continue
                
        print(f"✅ Found {len(completed_tasks)} already completed models. They will be skipped.")
        print("-" * 60)
        
    # Build the specific dictionary for each grid point
    grid_tasks = []
    
    for (m, tint, tirr, met, core, fsed, kzz) in combinations:
        
        current_tuple = (m, tint, tirr, met, core, fsed, kzz)
        rounded_current = tuple(round(float(v), 4) for v in current_tuple)
        
        # Check if this exact random point was already processed
        if rounded_current in completed_tasks:
            continue  
            
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
        print("🎉 All random grid combinations have already been successfully computed! Nothing to do.")
        exit(0)
        
    print(f"🚀 INITIALIZING EXOWEAVE RANDOMIZED GRID COMPUTING...")
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
                
            # Formatting to 3 decimals so the terminal output isn't a mess of long floats
            print(f"[{i}/{total_models}] {icon} M={res['mass']:.3f} | "
                  f"T_int={res['T_int']:.1f} | Core={res['core']:.2f} | Status: {res['status']} "
                  f"(Iters: {res['iterations']})")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"🏁 GRID COMPLETE IN {elapsed/60:.2f} MINUTES.")
    print(f"📊 Success Rate for this batch: {successful}/{total_models} ({(successful/total_models)*100:.1f}%)")
    print(f"💾 All converged models are safely stored in: {GRID_CONFIG['output_dir']}")