import pickle
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Stefan-Boltzmann constant (W m^-2 K^-4)
SIGMA_SB = 5.670374419e-8

def recalculate_dt_ds(results_dict: dict, parameters: dict) -> bool:
    """
    Extracts the interior arrays, masks out the core (S = 0), and recalculates 
    the cooling rates. Updates the dictionary in-place.
    Returns True if successful, False if missing data.
    """
    int_raw = results_dict.get('interior_raw')
    if not int_raw:
        return False
        
    required_keys = ['R', 'T', 'M', 'Z', 'S']
    if not all(k in int_raw for k in required_keys):
        return False

    r_array = int_raw['R']
    t_array = int_raw['T']
    m_array = int_raw['M']
    z_array = int_raw['Z']
    s_array = int_raw['S']
    
    # Extract the effective T_int (from coupler.py parameter outputs)
    t_int = parameters.get('T_int', parameters.get('T_surf', np.nan))
    radius_planet = r_array[-1]
    
    # Stefan-Boltzmann denominator
    denominator = SIGMA_SB * (radius_planet ** 2) * (t_int ** 4)
    if denominator <= 0:
        return False

    # Exact mass of each discrete spherical shell
    dm = np.diff(m_array)

    # Approximate average properties of the shell
    t_shell = (t_array[:-1] + t_array[1:]) / 2.0
    z_shell = z_array[1:]
    s_shell = s_array[1:]

    # 🚨 THE MASK: Only integrate shells where Entropy is strictly positive
    env_mask = s_shell > 0.0

    # Apply the mask to isolate envelope properties
    dm_env = dm[env_mask]
    t_shell_env = t_shell[env_mask]
    z_shell_env = z_shell[env_mask]

    # Evaluate the integrand strictly for the envelope: T(r) * (dm / 4*pi)
    integrand = t_shell_env * (dm_env / (4 * np.pi))
    unique_z = np.unique(z_shell_env)

    layer_contributions = {}
    total_dt_ds = 0.0

    # Integrate layer-by-layer
    for z_val in unique_z:
        mask = np.isclose(z_shell_env, z_val, atol=1e-4)
        layer_integral = np.sum(integrand[mask])
        layer_dt_ds = -(layer_integral / denominator)
        
        layer_contributions[z_val] = layer_dt_ds
        total_dt_ds += layer_dt_ds

    # Inject the corrected values back into the dictionary
    int_raw['dt_ds_total'] = total_dt_ds
    int_raw['dt_ds_layers'] = layer_contributions
    
    return True

def repair_all_pkls(target_dir: str):
    """Scans and patches all .pkl files in the target directory."""
    pkl_dir = Path(target_dir)
    if not pkl_dir.exists():
        logging.error(f"❌ Directory {target_dir} not found.")
        return

    target_files = list(pkl_dir.glob("target/*.pkl"))
    step_files = list(pkl_dir.glob("steps/*.pkl"))
    pkl_files = target_files + step_files
    
    if not pkl_files:
        logging.warning(f"⚠️ No files found in {pkl_dir}/target/ or {pkl_dir}/steps/")
        return

    logging.info(f"🔍 Scanning {len(target_files)} target and {len(step_files)} step files to repair cooling rates...")
    models_repaired = 0

    for idx, pkl_file in enumerate(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logging.warning(f"⚠️ Could not read {pkl_file.name}: {e}")
            continue
            
        # Get parameters safely
        params = data.get('final_params', data.get('parameters', {}))
        
        # Recalculate and update the dictionary in memory
        success = recalculate_dt_ds(data, params)
        
        if success:
            # Safely overwrite the .pkl file using the .tmp safeguard
            temp_file = pkl_file.with_suffix('.pkl.tmp')
            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump(data, f)
                temp_file.replace(pkl_file)
                models_repaired += 1
            except Exception as e:
                logging.error(f"❌ Failed to save updates to {pkl_file.name}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
        
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1}/{len(pkl_files)} files...")

    logging.info(f"\n✅ Successfully repaired cooling rates for {models_repaired} .pkl files!")
    logging.info("🚀 You can now run `compile_grid.py` to seamlessly rebuild your HDF5 and CSV catalogs with the corrected data.")

if __name__ == "__main__":
    GRID_OUTPUT_DIR = "../outputs/grid_run"
    repair_all_pkls(target_dir=GRID_OUTPUT_DIR)