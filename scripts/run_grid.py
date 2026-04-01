import os
import time
import logging
import pickle
import numpy as np
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import ExoWeave Coupler
from exoweave.coupler import ExoCoupler

# Suppress debug noise from individual processes
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# 1A. REGULAR GRID DEFINITION
# =============================================================================
# Define specific values to compute every possible permutation.
REGULAR_GRID = {
    "mass": [1.0],        # Jupiter Masses
    "T_int": np.arange(100, 1800, 10),    # Internal Temperatures (K)
    "T_irr": [100],        # Irradiation (K)
    "Met": [0.0],   # Metallicity (log10 Z/Z_solar)
    "core_mass_earth": [10.0],      # Solid core mass in Earth masses
    "f_sed": [3.0],                       # Cloud sedimentation
    "kzz": [8.0]                          # Eddy diffusion (log10)
}

# =============================================================================
# 1B. RANDOM GRID DEFINITION
# =============================================================================
PARAM_BOUNDS = {
    "mass": (0.05, 13.0),             
    "T_int": (1000.0, 1600.0),        
    "T_irr": (0.0, 1500.0),       
    "Met": (-2, 2),                 
    "core_mass_earth": (1, 100),    
    "f_sed": (1.0, 7.0),            
    "kzz": (0.0, 12.0)               
}

TOTAL_RANDOM_MODELS = 0          

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
    "resolution": 50,           
    "target_resolution": 500    
}

# =============================================================================
# 2. CACHE & WORKER FUNCTIONS
# =============================================================================
def get_cache_key(m, tint, tirr, met, core, fsed, kzz):
    """
    Creates a robust cache signature by binning T_int to the nearest 50 K.
    This prevents the pipeline from re-running models where T_int was 
    modified post-run by the RCB Sniper physics correction.
    """
    try:
        binned_tint = round(float(tint) / 50.0) * 50.0
        return (
            round(float(m), 4),
            binned_tint,
            round(float(tirr), 4),
            round(float(met), 4),
            round(float(core), 4),
            round(float(fsed), 4),
            round(float(kzz), 4)
        )
    except (TypeError, ValueError):
        return None

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
    
    combinations = []

    # --- Generate Regular Grid ---
    regular_combinations = list(product(
        REGULAR_GRID["mass"],
        REGULAR_GRID["T_int"],
        REGULAR_GRID["T_irr"],
        REGULAR_GRID["Met"],
        REGULAR_GRID["core_mass_earth"],
        REGULAR_GRID["f_sed"],
        REGULAR_GRID["kzz"]
    ))
    combinations.extend(regular_combinations)
    print(f"📐 Generated {len(regular_combinations)} regular grid points.")
    
    # --- Generate Random Grid ---
    print(f"🎲 Generating {TOTAL_RANDOM_MODELS}")    
    for _ in range(TOTAL_RANDOM_MODELS):
        m = np.random.uniform(*PARAM_BOUNDS["mass"])
        tint = np.random.uniform(*PARAM_BOUNDS["T_int"])
        tirr = np.random.uniform(*PARAM_BOUNDS["T_irr"])
        met = np.random.uniform(*PARAM_BOUNDS["Met"])
        core = np.random.uniform(*PARAM_BOUNDS["core_mass_earth"])
        fsed = np.random.uniform(*PARAM_BOUNDS["f_sed"])
        kzz = np.random.uniform(*PARAM_BOUNDS["kzz"])
        
        combinations.append((m, tint, tirr, met, core, fsed, kzz))
    
    # --- PRE-FLIGHT CACHE SCANNER ---
    output_dir = Path('./../' + GRID_CONFIG["output_dir"])
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
                    
                    # Try to get the dial first, fallback to the physical (sniper) T_int
                    target_tint = p.get('T_int_input_dial', p.get('T_int'))
                    
                    cache_key = get_cache_key(
                        p.get('mass'),
                        target_tint,
                        p.get('T_irr'),
                        p.get('Met'),
                        p.get('core_mass_earth'),
                        p.get('f_sed'),
                        p.get('kzz')
                    )
                    
                    if cache_key is not None:
                        completed_tasks.add(cache_key)
            except Exception:
                print(f"⚠️ Warning: Could not read {pkl_file.name}. Skipping this file.")
                continue
                
        print(f"✅ Found {len(completed_tasks)} already completed models. They will be skipped.")
        print("-" * 60)
        
    # Build the specific dictionary for each grid point
    grid_tasks = []
    
    for combo in combinations:
        
        # Check against the robust 50K-binned cache key
        cache_key = get_cache_key(*combo)
        
        if cache_key in completed_tasks:
            continue  
            
        params = STATIC_PARAMS.copy()
        params.update({
            "mass": combo[0],
            "T_int": combo[1],
            "T_irr": combo[2],
            "Met": combo[3],
            "core_mass_earth": combo[4],
            "f_sed": combo[5],
            "kzz": combo[6]
        })
        grid_tasks.append(params)
        
    total_models = len(grid_tasks)
    
    if total_models == 0:
        print("🎉 All grid combinations have already been successfully computed! Nothing to do.")
        exit(0)
        
    print(f"🚀 INITIALIZING EXOWEAVE COMBINED GRID COMPUTING...")
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
                
            print(f"[{i}/{total_models}] {icon} M={res['mass']:.3f} | "
                  f"T_int={res['T_int']:.1f} | Core={res['core']:.2f} | Status: {res['status']} "
                  f"(Iters: {res['iterations']})")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"🏁 GRID COMPLETE IN {elapsed/60:.2f} MINUTES.")
    print(f"📊 Success Rate for this batch: {successful}/{total_models} ({(successful/total_models)*100:.1f}%)")
    print(f"💾 All converged models are safely stored in: {GRID_CONFIG['output_dir']}")