import os
import pickle
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# Set up clean logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import numpy as np
import pandas as pd

class ExoFilter:
    """Quality Control filter using exact ExoWeave .pkl and HDF5 keys."""
    
    @staticmethod
    def check_pt_continuity(prof_df, catalog_entry, min_jump_k=150, grad_threshold=5000):
        """
        Intelligently detects bad stitching in the envelope by evaluating the 
        thermodynamic gradient, while explicitly ignoring the degenerate core.
        """
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Temperature_K' not in prof_df.columns or 'Pressure_Pa' not in prof_df.columns:
            return False, "MISSING_COLUMNS"
            
        # 1. Anchor the search to the stitch point. Default to 1000 bar if missing.
        p_link_bar = catalog_entry.get('P_link_bar', 1000.0)
        if np.isnan(p_link_bar): 
            p_link_bar = 1000.0
            
        # 2. Define the safe zone: Check the atmosphere and the upper envelope.
        # Stopping 1,000x below the link pressure ensures we check the stitch 
        # but stop well before the core boundary (which is usually > 10 Mbar).
        max_valid_pressure_pa = p_link_bar * 1e5 * 1000.0
        
        mask = prof_df['Pressure_Pa'].values < max_valid_pressure_pa
        
        t_values = prof_df.loc[mask, 'Temperature_K'].values
        p_values = prof_df.loc[mask, 'Pressure_Pa'].values
        
        if len(t_values) < 2:
            return True, "OK" # Fallback if mask removes everything
            
        # 3. Calculate layer-to-layer differences
        dt = np.abs(np.diff(t_values))
        dp_log = np.abs(np.diff(np.log10(p_values)))
        dp_log = np.maximum(dp_log, 1e-8) # Prevent division by zero
        
        # 4. Calculate local gradient (Kelvin per pressure decade)
        gradients = dt / dp_log
        
        # 5. Identify the most severe discontinuity in the ENVELOPE
        max_idx = np.argmax(gradients)
        worst_grad = gradients[max_idx]
        worst_jump = dt[max_idx]
        median_jump = np.median(dt)
        
        # 6. The Clever Check:
        # A bad stitch is a sharp spike, a statistical outlier, and > 150K.
        if worst_jump > min_jump_k and worst_jump > (5 * median_jump) and worst_grad > grad_threshold:
            return False, f"PT_DISCONT_{worst_jump:.0f}K_grad{worst_grad:.1e}"
            
        return True, "OK" 
    
    @staticmethod
    def check_pressure_resolution(prof_df, max_dp_log=0.5):
        """
        Ensures the profile does not have massive gaps in the pressure grid.
        Flags any grid where adjacent pressure points jump by more than the allowed limit.
        """
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Pressure_Pa' not in prof_df.columns:
            return False, "MISSING_PRESSURE_COLUMN"
            
        p_values = prof_df['Pressure_Pa'].values
        
        # Calculate the absolute step size in log10(Pressure)
        dp_log = np.abs(np.diff(np.log10(p_values)))
        
        # Find the single largest gap in the entire grid
        max_step = np.max(dp_log)
        
        if max_step > max_dp_log:
            return False, f"GRID_GAP_{max_step:.2f}dex"
            
        return True, "OK"

    @staticmethod
    def check_physical_radius(r_total_m):
        """Flags planets with highly unphysical total radii."""
        if r_total_m is None or np.isnan(r_total_m):
            return False, "NO_RADIUS_DATA"
            
        # Convert meters to Jupiter Radii
        r_jup = r_total_m / 71492000.0
        
        # Flag if radius is smaller than 0.3 Rjup or larger than 4.0 Rjup
        if r_jup < 0.3 or r_jup > 4.0:
            return False, f"UNPHYSICAL_RADIUS_{r_jup:.2f}Rj"
            
        return True, "OK"

    @staticmethod
    def check_cooling_rate(int_raw):
        """Checks interior_raw for a valid dt_ds_total."""
        if not int_raw:
            return False, "NO_INTERIOR_DATA"
            
        dt_ds = int_raw.get('dt_ds_total', 0.0)
        
        # If it's exactly 0.0 or NaN, the cooling solver failed
        if dt_ds == 0.0 or np.isnan(dt_ds):
            return False, "FLAT_COOLING"
            
        return True, "OK"
    
    @staticmethod
    def check_pt_percentage_jump(prof_df, threshold_pct=100.0):
        """
        Flags profiles where the temperature jumps by more than a specified 
        percentage relative to the adjacent layer.
        """
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Temperature_K' not in prof_df.columns:
            return False, "MISSING_TEMP_COLUMN"
            
        t_values = prof_df['Temperature_K'].values
        
        if len(t_values) < 2:
            return True, "OK"
            
        dt = np.abs(np.diff(t_values))
        t_prev = t_values[:-1]
        
        # Guard against zero-division just in case
        t_prev_safe = np.maximum(t_prev, 1e-8)
        
        dt_percent = (dt / t_prev_safe) * 100.0
        
        max_idx = np.argmax(dt_percent)
        max_pct_val = dt_percent[max_idx]
        
        if max_pct_val > threshold_pct:
            return False, f"PT_JUMP_{max_pct_val:.0f}PCT"
            
        return True, "OK"

    @staticmethod
    def check_max_tint(t_int_true_k, max_t_int=2000.0):
        """
        Flags models where the true internal temperature exceeds the physical 
        validity of the opacity tables.
        """
        if t_int_true_k is None or np.isnan(t_int_true_k):
            return False, "NO_TINT_DATA"
            
        if t_int_true_k > max_t_int:
            return False, f"TINT_TOO_HIGH_{t_int_true_k:.0f}K"
            
        return True, "OK"

    @staticmethod
    def validate(data, catalog_entry):
        """Main validation router."""
        if data.get('status') == 'failed' or 'failure_reason' in data:
            return False, "SOLVER_FAILED"

        prof_df = data.get('stitched_profile') if 'stitched_profile' in data else data.get('profile')
        int_raw = data.get('interior_raw', {})
        r_total_m = int_raw.get('R_total', catalog_entry.get('R_total_m'))
        t_int_true = catalog_entry.get('T_int_true_K') # Get the true T_int

        # 1. Grid Resolution Check
        grid_ok, grid_reason = ExoFilter.check_pressure_resolution(prof_df)
        if not grid_ok: return False, grid_reason

        # 2. Relative Percentage Jump Check
        pct_ok, pct_reason = ExoFilter.check_pt_percentage_jump(prof_df, threshold_pct=100.0)
        if not pct_ok: return False, pct_reason

        # 3. Absolute Gradient Check
        #pt_ok, pt_reason = ExoFilter.check_pt_continuity(prof_df, catalog_entry)
        #if not pt_ok: return False, pt_reason
        
        # 4. Physical Radius Check
        rad_ok, rad_reason = ExoFilter.check_physical_radius(r_total_m)
        if not rad_ok: return False, rad_reason
        
        # 5. NEW: Maximum Valid Temperature Check
        tint_ok, tint_reason = ExoFilter.check_max_tint(t_int_true, max_t_int=2000.0)
        if not tint_ok: return False, tint_reason
        
        # 6. Cooling Rate Check
        cool_ok, cool_reason = ExoFilter.check_cooling_rate(int_raw)
        if not cool_ok: return False, cool_reason
        
        return True, "VALID"
    

def compile_exoweave_grid(input_dir: str, output_prefix: str):
    """
    Scans a directory of ExoWeave .pkl outputs and compiles them into:
      1. A searchable CSV catalog (_catalog.csv)
      2. A highly compressed master HDF5 binary data store (_data.h5)
      3. A lightweight CoolTrack-specific HDF5 extract (_cooltrack.h5)
    """
    in_path = Path(input_dir)
    if not in_path.exists():
        logging.error(f"❌ Directory {input_dir} not found!")
        return

    csv_path = Path(f"{output_prefix}_catalog.csv")
    h5_path = Path(f"{output_prefix}_data.h5")
    cooltrack_path = Path(f"{output_prefix}_cooltrack.h5")
    
    pkl_files = list(in_path.glob("**/*.pkl"))
    pkl_files.sort()
    
    if len(pkl_files) == 0:
        logging.error("❌ No .pkl files found to compile.")
        return

    summary_catalog = []
    total_files = len(pkl_files)
    logging.info(f"🚀 Starting compilation of {total_files} models...")

    # Open both HDF5 files in append mode to build the grid
    with h5py.File(h5_path, 'w') as h5_master, h5py.File(cooltrack_path, 'w') as h5_cool:
        
        for idx, pkl_file in enumerate(pkl_files):
            model_id = f"model_{idx:05d}"
            
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                logging.warning(f"⚠️ Could not read {pkl_file.name}: {e}")
                continue

            # =================================================================
            # 1. STATUS ROUTING
            # =================================================================
            if 'failed' in pkl_file.parts:
                status = 'crashed'
            else:
                raw_status = data.get('status', 'converged')
                if raw_status == 'converged':
                    status = 'target_reached'
                elif raw_status == 'intermediate':
                    status = 'intermediate_step'
                else:
                    status = 'max_iterations_reached'

            # =================================================================
            # 2. BASE CATALOG ENTRY
            # =================================================================
            params = data.get('final_params', data.get('parameters', {}))
            iters = data.get('iterations', data.get('iteration', np.nan))

            catalog_entry = {
                'model_id': model_id,
                'status': status,
                'target_mass_Mjup': params.get('mass', np.nan),
                'true_mass_Mjup': params.get('true_mass_Mjup', np.nan),
                'T_int_dial_K': params.get('T_int_input_dial', params.get('T_int', np.nan)),
                'T_int_true_K': params.get('T_int', np.nan),
                'T_irr_K': params.get('T_irr', np.nan),
                'metallicity': params.get('Met', np.nan),
                'core_mass_Me': params.get('core_mass_earth', np.nan),
                'f_sed': params.get('f_sed', np.nan),
                'kzz': params.get('kzz', np.nan),
                'iterations': iters, 
                'P_link_bar': params.get('p_link_bar', np.nan),
                'R_total_m': np.nan,      
                'R_1bar_Rjup': np.nan,    
                'original_file': pkl_file.name
            }

            # =================================================================
            # 3. METRIC EXTRACTION (Runs for any model with data)
            # =================================================================
            prof_df = data.get('profile') if 'profile' in data else data.get('stitched_profile')
            atm_df = data.get('atmosphere_raw')
            int_raw = data.get('interior_raw', {})
            phot_data = data.get('photometry', {})

            # 3a. Radius Interpolation
            if prof_df is not None and 'Pressure_bar' in prof_df.columns and 'Radius_m' in prof_df.columns:
                try:
                    sorted_prof = prof_df.sort_values('Pressure_bar')
                    r_1bar_m = np.interp(1.0, sorted_prof['Pressure_bar'], sorted_prof['Radius_m'])
                    catalog_entry['R_1bar_Rjup'] = r_1bar_m / 71492000.0
                    
                    if int_raw and 'R_total' in int_raw:
                        catalog_entry['R_total_m'] = int_raw['R_total']
                except Exception as e:
                    logging.debug(f"Could not interpolate radius for {model_id}: {e}")

            # 3b. Photometry Flattening
            if phot_data and 'bands' in phot_data:
                for filt_id, metrics in phot_data['bands'].items():
                    safe_filt_id = filt_id.replace('/', '_')
                    if 'flux_W_m2_um' in metrics:
                        catalog_entry[f"{safe_filt_id}_flux_Wm2um"] = metrics['flux_W_m2_um']
                    if 'flux_Jy' in metrics:
                        catalog_entry[f"{safe_filt_id}_flux_Jy"] = metrics['flux_Jy']

            # =================================================================
            # 4. RUN QUALITY CONTROL FILTER
            # =================================================================
            # NOTE: Assumes you have the ExoFilter class defined above this function
            is_valid, reason = ExoFilter.validate(data, catalog_entry)
            catalog_entry['qc_status'] = reason
            
            summary_catalog.append(catalog_entry)

            # If it failed physics checks or was a hard crash, SKIP binary export
            if not is_valid:
                logging.warning(f"⚠️ {model_id} failed QC: {reason}. Skipping binary export.")
                continue 

            # =================================================================
            # 5. HDF5 MASTER & COOLTRACK EXPORT (Runs for all VALID models)
            # =================================================================
            
            # --- 5a. Master HDF5 File ---
            model_grp = h5_master.create_group(model_id)
            
            # Parameters
            param_grp = model_grp.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str, bytes, bool)):
                    param_grp.attrs[k] = v

            # Interior Arrays
            if int_raw:
                int_grp = model_grp.create_group('interior_raw')
                for k, v in int_raw.items():
                    if isinstance(v, (int, float, str, bytes, bool)):
                        int_grp.attrs[k] = v
                    elif isinstance(v, (np.ndarray, list)):
                        int_grp.create_dataset(k, data=np.array(v))
                        
            # Stitched Profile
            if prof_df is not None and not prof_df.empty:
                prof_grp = model_grp.create_group('stitched_profile')
                for col in prof_df.columns:
                    prof_grp.create_dataset(col, data=prof_df[col].values)
            
            # Photometry
            if phot_data and 'bands' in phot_data:
                phot_grp = model_grp.create_group('photometry')
                
                # Save raw spectra if present
                if 'wavelength_um' in phot_data and 'emission_flux_W_m2_um' in phot_data:
                    phot_grp.create_dataset('wavelength_um', data=phot_data['wavelength_um'])
                    phot_grp.create_dataset('emission_flux_W_m2_um', data=phot_data['emission_flux_W_m2_um'])
                
                # Save integrated bands
                bands_grp = phot_grp.create_group('bands')
                for filt_id, metrics in phot_data['bands'].items():
                    safe_filt_id = filt_id.replace('/', '_')
                    f_grp = bands_grp.create_group(safe_filt_id)
                    f_grp.attrs['filter_id'] = filt_id
                    for k, v in metrics.items():
                        if k != 'filter_id': # already saved
                            f_grp.attrs[k] = v

            # --- 5b. CoolTrack Sub-Extract ---
            # Lightweight extract specifically for cooling track algorithms
            ct_grp = h5_cool.create_group(model_id)
            
            # Parameters
            ct_param = ct_grp.create_group('parameters')
            for key in ['mass', 'true_mass_Mjup', 'T_int', 'T_irr', 'Met', 'core_mass_earth', 'f_sed', 'kzz']:
                if key in params:
                    ct_param.attrs[key] = params[key]
                        
            if int_raw:
                ct_int = ct_grp.create_group('interior_raw')
                for key in ['dt_ds_total', 'M_total', 'R_total', 'S']:
                    if key in int_raw:
                        ct_int.attrs[key] = int_raw[key]

            if phot_data and 'bands' in phot_data:
                ct_phot = ct_grp.create_group('photometry')
                ct_bands = ct_phot.create_group('bands')
                for filt_id, metrics in phot_data['bands'].items():
                    safe_filt_id = filt_id.replace('/', '_')
                    f_grp = ct_bands.create_group(safe_filt_id)
                    if 'flux_W_m2_um' in metrics:
                        f_grp.attrs['flux_W_m2_um'] = metrics['flux_W_m2_um']

            # Print progress cleanly
            if (idx + 1) % 50 == 0:
                logging.info(f"Processed {idx + 1}/{total_files} files...")

    # =================================================================
    # 6. SAVE CSV CATALOG
    # =================================================================
    df_catalog = pd.DataFrame(summary_catalog)
    
    # Reorder columns to keep the basics at the front
    cols = df_catalog.columns.tolist()
    if 'R_1bar_Rjup' in cols:
        cols.insert(3, cols.pop(cols.index('R_1bar_Rjup')))
    df_catalog = df_catalog[cols]
    
    df_catalog.to_csv(csv_path, index=False)
    
    logging.info(f"✅ Grid Compilation Complete!")
    logging.info(f"📊 Catalog saved to: {csv_path}")
    logging.info(f"🗄️ Master Data stored in: {h5_path}")
    logging.info(f"🧊 CoolTrack Extract stored in: {cooltrack_path}")

if __name__ == "__main__":
    TARGET_GRID_DIR = "../outputs/grid_run"
    OUTPUT_PREFIX = "../outputs/master_grid"
    compile_exoweave_grid(TARGET_GRID_DIR, OUTPUT_PREFIX)