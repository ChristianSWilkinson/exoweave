import os
import time
import logging
import pickle
import random
import numpy as np
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import qmc  

# Import ExoWeave Coupler
from exoweave.coupler import ExoCoupler

# Suppress debug noise from individual processes
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Explicit list of active clouds
ACTIVE_CLOUDS = ["Fe", "Mg2SiO4", "H2O", "NH3", "NH4SH"]

# =============================================================================
# 1A. REGULAR GRID DEFINITION
# =============================================================================
REGULAR_GRID = {
    "mass": np.arange(0.2, 10.2, 0.2),          # Jupiter Masses
    "T_int": np.arange(100, 1710, 10),          # Internal Temperatures (K)
    "T_irr": [0, 100, 500, 1000, 1500],         # Irradiation (K)
    "Met": [0],                                 # Metallicity (log10 Z/Z_solar)
    "core_mass_earth": [10],                    # Solid core mass in Earth masses
    "kzz": [8],                                 # Eddy diffusion (log10)
    "f_sed_refractory": [0.5, 1, 2, 3, 4, 5, 6], # Controls Fe, Mg2SiO4
    "f_sed_volatile":   [0.5, 1, 2, 3, 4, 5, 6]  # Controls H2O, NH3, NH4SH
}

# =============================================================================
# 1B. RANDOM GRID DEFINITION (BOUNDS)
# =============================================================================
PARAM_BOUNDS = {
    "mass": (0.05, 13.0),             
    "T_int": (0.0, 1600.0),        
    "T_irr": (0.0, 1500.0),       
    "Met": (-2.1, 2.1),                 
    "core_mass_earth": (1, 100),    
    "kzz": (0.0, 12.0),
    "f_sed_refractory": (0.5, 7.0),            
    "f_sed_volatile": (0.5, 7.0)               
}

TOTAL_RANDOM_MODELS = 500  

# =============================================================================
# 1C. DIRECTORY CONFIGURATION & STATIC PARAMS
# =============================================================================
# The directory containing your OLD models (to warm-start the P-T profiles)
SOURCE_GRID_DIR = Path("../outputs/grid_run/target")

GRID_CONFIG = {
    "output_dir": "outputs/grid_run_5clouds",  
    "prior_dir": str(SOURCE_GRID_DIR),          # <--- RESTORED WARM-START DIRECTORY
    "max_iterations": 15,
    "mass_convergence_threshold": 0.01,
    "p_bottom_bar": 1000.0,
    "resolution": 50,           
    "target_resolution": 500,    
    "retrieval_flux_error_bottom": 1e-4,
    "retrieval_flux_error_top": 1e-4,
    "n_iterations": 101,                      
    "n_non_adiabatic_iterations": 0,
    "chemistry_iteration_interval": 5,
    "cloud_iteration_interval": 0,
    "n_burn_iterations": 0,
    "retrieval_tolerance": 0.0,
    "smoothing_bottom": 0.5,
    "smoothing_top": 0.5,
    "weight_apriori": 0.001
}

STATIC_PARAMS = {
    "active_clouds": ACTIVE_CLOUDS,
    "iron_fraction": 0.33,
    "debug": False  
}

# =============================================================================
# 2. CACHE & WORKER FUNCTIONS
# =============================================================================
def get_cache_key(m, tint, tirr, met, core, fsed, kzz):
    """Creates a robust cache signature extracting Refractory and Volatile f_sed values."""
    try:
        binned_tint = round(float(tint) / 50.0) * 50.0
        
        if isinstance(fsed, dict):
            fsed_ref = round(float(fsed.get("Fe", 6.0)), 2)
            fsed_vol = round(float(fsed.get("H2O", 6.0)), 2)
        else:
            fsed_ref = round(float(fsed), 2)
            fsed_vol = fsed_ref

        return (
            round(float(m), 4),
            binned_tint,
            round(float(tirr), 4),
            round(float(met), 4),
            round(float(core), 4),
            fsed_ref,      
            fsed_vol,      
            round(float(kzz), 4)
        )
    except (TypeError, ValueError):
        return None

def run_model(target_params: dict) -> dict:
    try:
        time.sleep(random.uniform(0.1, 1.0))
        coupler = ExoCoupler(target_params=target_params, config=GRID_CONFIG)
        results = coupler.run()
        
        return {
            "mass": target_params.get("mass", 0),
            "T_int": target_params.get("T_int", 0),
            "core": target_params.get("core_mass_earth", 0),
            "f_ref": target_params.get("f_sed", {}).get("Fe", 0),
            "f_vol": target_params.get("f_sed", {}).get("H2O", 0),
            "status": results["status"],
            "iterations": results.get("iterations", "N/A"),
        }
    except Exception as e:
        return {
            "mass": target_params.get("mass", 0),
            "T_int": target_params.get("T_int", 0),
            "core": target_params.get("core_mass_earth", 0),
            "f_ref": target_params.get("f_sed", {}).get("Fe", 0),  # <-- ADD THIS
            "f_vol": target_params.get("f_sed", {}).get("H2O", 0), # <-- ADD THIS
            "status": f"crashed: {str(e)}",
            "iterations": "N/A" 
        }

# =============================================================================
# 3. MAIN EXECUTION POOL
# =============================================================================
if __name__ == "__main__":
    
    grid_tasks = []

    # --- A. Generate Regular Grid ---
    regular_combinations = list(product(
        REGULAR_GRID["mass"], REGULAR_GRID["T_int"], REGULAR_GRID["T_irr"],
        REGULAR_GRID["Met"], REGULAR_GRID["core_mass_earth"], 
        REGULAR_GRID["f_sed_refractory"], REGULAR_GRID["f_sed_volatile"], 
        REGULAR_GRID["kzz"]
    ))
    
    for combo in regular_combinations:
        p = STATIC_PARAMS.copy()
        f_sed_dict = {
            "Fe": combo[5], "Mg2SiO4": combo[5],
            "H2O": combo[6], "NH3": combo[6], "NH4SH": combo[6]
        }
        
        p.update({
            "mass": combo[0], "T_int": combo[1], "T_irr": combo[2],
            "Met": combo[3], "core_mass_earth": combo[4], 
            "f_sed": f_sed_dict, "kzz": combo[7]
        })
        grid_tasks.append(p)
    print(f"📐 Added {len(regular_combinations)} regular grid points.")
    
    # --- B. Generate Random Grid using Latin Hypercube Sampling (LHS) ---
    if TOTAL_RANDOM_MODELS > 0:
        # 1. Define the parameters we are sampling
        param_keys = ["mass", "T_int", "T_irr", "Met", "core_mass_earth", 
                      "kzz", "f_sed_refractory", "f_sed_volatile"]
        
        # 2. Extract lower and upper bounds in the exact same order
        l_bounds = []
        u_bounds = []
        for k in param_keys:
            if k == "mass":
                l_bounds.append(np.log10(PARAM_BOUNDS[k][0]))
                u_bounds.append(np.log10(PARAM_BOUNDS[k][1]))
            else:
                l_bounds.append(PARAM_BOUNDS[k][0])
                u_bounds.append(PARAM_BOUNDS[k][1])
        
        # 3. Initialize the LHS sampler
        sampler = qmc.LatinHypercube(d=len(param_keys))
        
        # 4. Generate points in a unit hypercube [0, 1]
        lhs_unit_samples = sampler.random(n=TOTAL_RANDOM_MODELS)
        
        # 5. Scale the unit samples to our physical parameter bounds
        scaled_samples = qmc.scale(lhs_unit_samples, l_bounds, u_bounds)
        
        # 6. Unpack and append to grid tasks
        for i in range(TOTAL_RANDOM_MODELS):
            p = STATIC_PARAMS.copy()
            log_m, t_int, t_irr, met, core, kzz, f_ref, f_vol = scaled_samples[i]
            
            linear_mass = 10 ** log_m
            
            random_fsed_dict = {
                "Fe": f_ref, "Mg2SiO4": f_ref,
                "H2O": f_vol, "NH3": f_vol, "NH4SH": f_vol
            }
            
            p.update({
                "mass": linear_mass, "T_int": t_int, "T_irr": t_irr,
                "Met": met, "core_mass_earth": core,
                "f_sed": random_fsed_dict, "kzz": kzz
            })
            grid_tasks.append(p)
        print(f"🎲 Added {TOTAL_RANDOM_MODELS} LHS points (Mass sampled in log-space).")
    
    # --- D. GRID CACHE SCANNER ---
    new_output_dir = Path('./../' + GRID_CONFIG["output_dir"])
    completed_tasks = set()
    
    if new_output_dir.exists():
        for pkl_file in new_output_dir.glob("**/*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                if data.get('status') in ['converged', 'max_iterations_reached']:
                    p = data.get('final_params', data.get('parameters', {}))
                    cache_key = get_cache_key(
                        p.get('mass'), p.get('T_int_input_dial', p.get('T_int')),
                        p.get('T_irr'), p.get('Met'), p.get('core_mass_earth'),
                        p.get('f_sed'), p.get('kzz')
                    )
                    if cache_key is not None:
                        completed_tasks.add(cache_key)
            except Exception:
                continue
        print(f"✅ Found {len(completed_tasks)} already completed 5-cloud models. They will be skipped.")

    # Filter out already completed tasks
    final_tasks = []
    for task in grid_tasks:
        key = get_cache_key(
            task['mass'], task['T_int'], task['T_irr'], 
            task['Met'], task['core_mass_earth'], task['f_sed'], task['kzz']
        )
        if key not in completed_tasks:
            completed_tasks.add(key)
            final_tasks.append(task)

    total_models = len(final_tasks)
    if total_models == 0:
        print("🎉 All grid combinations have already been successfully computed! Nothing to do.")
        exit(0)

    print("🔀 Shuffling tasks to interleave regular and LHS grids...")
    random.shuffle(final_tasks)
        
    print(f"\n🚀 INITIALIZING EXOWEAVE 5-CLOUD GRID COMPUTING...")
    print(f"📦 Unique Models to Compute: {total_models}")
    print(f"📂 Outputting new models to: {GRID_CONFIG['output_dir']}")
    print(f"💻 CPU Cores Detected: {os.cpu_count()}")
    print("-" * 60)

    start_time = time.time()
    max_cores = max(1, os.cpu_count() - 2) 
    successful, failed = 0, 0
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        future_to_params = {executor.submit(run_model, p): p for p in final_tasks}
        
        for i, future in enumerate(as_completed(future_to_params), 1):
            res = future.result()
            if res["status"] == "converged":
                successful += 1
                icon = "✅"
            else:
                failed += 1
                icon = "❌"
                
            # --- UPDATED PRINT STATEMENT INSIDE THE LOOP ---
            print(f"[{i}/{total_models}] {icon} M={res['mass']:.2f} | "
                  f"T_int={res['T_int']:.0f} | f_ref={res['f_ref']:.2f} | "
                  f"f_vol={res['f_vol']:.2f} | Status: {res['status']} "
                  f"(Iters: {res['iterations']})")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"🏁 NEW CLOUD GRID COMPLETE IN {elapsed/60:.2f} MINUTES.")
    print(f"📊 Success Rate for this batch: {successful}/{total_models} ({(successful/total_models)*100:.1f}%)")