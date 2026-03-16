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
        pt_ok, pt_reason = ExoFilter.check_pt_continuity(prof_df, catalog_entry)
        if not pt_ok: return False, pt_reason
        
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
        logging.error(f"Directory {input_dir} not found!")
        return

    csv_path = Path(f"{output_prefix}_catalog.csv")
    h5_path = Path(f"{output_prefix}_data.h5")
    cooltrack_path = Path(f"{output_prefix}_cooltrack.h5") # <-- NEW: Cooltrack output
    
    pkl_files = list(in_path.glob("**/*.pkl"))
    # Sort files to ensure consistent ordering across runs
    pkl_files.sort()
    
    if len(pkl_files) == 0:
        logging.error("No .pkl files found to compile.")
        return

    # --- Check for Existing Progress to Resume ---
    processed_files = set()
    existing_catalog = []
    next_model_idx = 0
    
    if csv_path.exists() and h5_path.exists():
        logging.info("♻️ Existing grid found! Loading progress to resume...")
        try:
            df_existing = pd.read_csv(csv_path)
            if not df_existing.empty:
                # Store the filenames we've already done
                processed_files = set(df_existing['original_file'].dropna().values)
                # Keep the old data to merge with the new
                existing_catalog = df_existing.to_dict('records')
                
                # Find the highest model_id so we don't overwrite HDF5 folders
                if 'model_id' in df_existing.columns:
                    last_model_id = df_existing['model_id'].iloc[-1]
                    # Parse 'model_00142' -> 142 + 1 = 143
                    next_model_idx = int(last_model_id.split('_')[1]) + 1
        except Exception as e:
            logging.warning(f"⚠️ Could not read existing catalog ({e}). Starting fresh.")

    # Filter out files we've already processed
    files_to_process = [f for f in pkl_files if f.name not in processed_files]
    total_files = len(files_to_process)
    
    if total_files == 0:
        logging.info(f"✅ All {len(pkl_files)} files are already compiled. Nothing to do!")
        return

    logging.info(f"📦 Found {len(pkl_files)} total files. {len(processed_files)} already processed.")
    logging.info(f"🚀 Compiling {total_files} remaining models...")

    # Start with the existing catalog list so we don't lose old data
    summary_catalog = existing_catalog.copy()
    R_JUPITER_M = 71492000.0

    # --- Open both Master and Cooltrack HDF5 files in append mode ---
    with h5py.File(h5_path, 'a') as h5f, h5py.File(cooltrack_path, 'a') as ct_h5:
        
        for idx, pkl_file in enumerate(files_to_process):
            model_id = f"model_{next_model_idx:05d}"
            next_model_idx += 1  # Increment safely
            
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to read {pkl_file.name}: {e}")
                continue

            params = data.get('final_params', data.get('parameters', {}))
            
            # --- 1. Status Logic ---
            if 'failed' in pkl_file.parts:
                status = 'crashed'
            else:
                raw_status = data.get('status', 'converged')
                if raw_status == 'converged':
                    status = 'target_reached'
                elif raw_status == 'intermediate':
                    status = 'intermediate_step'  # Explicitly label them!
                else:
                    status = 'max_iterations_reached'

            # --- 2. Build the Base Catalog Entry ---
            # Look for both plural and singular keys
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
                'iterations': iters, # Use the safely extracted value
                'P_link_bar': params.get('p_link_bar', np.nan),
                'R_total_m': np.nan,      
                'R_1bar_Rjup': np.nan,    
                'original_file': pkl_file.name
            }

            is_valid, reason = ExoFilter.validate(data, catalog_entry)
            catalog_entry['qc_status'] = reason

            if not is_valid:
                logging.warning(f"⚠️ Model {model_id} failed QC: {reason}. Skipping binary export.")
                summary_catalog.append(catalog_entry)
                continue # Skip writing to HDF5, but keep in CSV for tracking

            # --- 3. Extract Arrays, Interpolate Radius, and Pull Photometry ---
            prof_df = data.get('profile') if 'profile' in data else data.get('stitched_profile')
            int_raw = data.get('interior_raw', {})
            phot_data = data.get('photometry', {})
            atm_df = data.get('atmosphere_raw')

            # ---------------------------------------------------------
            # Extract scalar metrics for ANY model that has data
            # ---------------------------------------------------------
            
            # 1. Radius Interpolation
            if prof_df is not None and 'Pressure_bar' in prof_df.columns and 'Radius_m' in prof_df.columns:
                try:
                    # Sort by pressure just to be safe for interpolation
                    sorted_prof = prof_df.sort_values('Pressure_bar')
                    
                    # Interpolate the radius at exactly 1.0 bar
                    r_1bar_m = np.interp(1.0, sorted_prof['Pressure_bar'], sorted_prof['Radius_m'])
                    catalog_entry['R_1bar_Rjup'] = r_1bar_m / 71492000.0
                    
                    # Grab total radius from interior if available
                    if int_raw and 'R_total' in int_raw:
                        catalog_entry['R_total_m'] = int_raw['R_total']
                except Exception as e:
                    logging.debug(f"Could not interpolate radius for {model_id}: {e}")

            # 2. Photometry Extraction
            if phot_data and 'bands' in phot_data:
                for filt_id, metrics in phot_data['bands'].items():
                    safe_filt_id = filt_id.replace('/', '_')
                    if 'flux_W_m2_um' in metrics:
                        catalog_entry[f"{safe_filt_id}_flux_Wm2um"] = metrics['flux_W_m2_um']
                    if 'flux_Jy' in metrics:
                        catalog_entry[f"{safe_filt_id}_flux_Jy"] = metrics['flux_Jy']

            # =================================================================
            # RUN QUALITY CONTROL FILTER (Happens AFTER we extract the radius)
            # =================================================================
            is_valid, reason = ExoFilter.validate(data, catalog_entry)
            catalog_entry['qc_status'] = reason
            
            if not is_valid:
                logging.warning(f"⚠️ Model {model_id} failed QC: {reason}. Skipping binary export.")
                summary_catalog.append(catalog_entry)
                continue 
            
            # --- 4. Populate the Master HDF5 Structure ---
            if model_id in h5f:
                del h5f[model_id]
            model_grp = h5f.create_group(model_id)

            param_grp = model_grp.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (str, bytes, int, float, bool, np.number)):
                    param_grp.attrs[k] = v

            if status in ['target_reached', 'max_iterations_reached']:
                
                # Master: Stitched Profile
                if prof_df is not None and not prof_df.empty:
                    prof_grp = model_grp.create_group('stitched_profile')
                    for col in prof_df.columns:
                        prof_grp.create_dataset(col, data=prof_df[col].values, compression="gzip")

                # Master: Atmosphere Raw
                if atm_df is not None and not atm_df.empty:
                    atm_grp = model_grp.create_group('atmosphere_raw')
                    for col in atm_df.columns:
                        val = atm_df[col].iloc[0]
                        safe_col = col.lstrip('/')
                        if isinstance(val, (str, bytes, int, float, bool, np.number)):
                            atm_grp.attrs[safe_col] = val
                        else:
                            try:
                                atm_grp.create_dataset(safe_col, data=np.asarray(val), compression="gzip")
                            except Exception:
                                pass

                # Master: Interior Raw
                if int_raw:
                    int_grp = model_grp.create_group('interior_raw')
                    for k, v in int_raw.items():
                        if isinstance(v, (str, bytes, int, float, bool, np.number)):
                            int_grp.attrs[k] = v
                        elif isinstance(v, (list, np.ndarray)):
                            try:
                                int_grp.create_dataset(k, data=np.asarray(v), compression="gzip")
                            except Exception:
                                pass
                                
                # Master: Photometry
                if phot_data:
                    phot_grp = model_grp.create_group('photometry')
                    for arr_key in ['wavelength_um', 'emission_flux_W_m2_um', 'transit_depth']:
                        arr_val = phot_data.get(arr_key)
                        if arr_val is not None:
                            try:
                                phot_grp.create_dataset(arr_key, data=np.asarray(arr_val), compression="gzip")
                            except Exception:
                                pass
                                
                    bands = phot_data.get('bands', {})
                    if bands:
                        bands_grp = phot_grp.create_group('bands')
                        for filt_id, metrics in bands.items():
                            safe_filt_id = filt_id.replace('/', '_')
                            f_grp = bands_grp.create_group(safe_filt_id)
                            for m_name, m_val in metrics.items():
                                if isinstance(m_val, (int, float, str, bool, np.number)):
                                    f_grp.attrs[m_name] = m_val

                # =================================================================
                # 5. Populate the COOLTRACK Minimal HDF5 Extract
                # =================================================================
                if model_id in ct_h5:
                    del ct_h5[model_id]
                ct_grp = ct_h5.create_group(model_id)
                
                # Cooltrack: Basic independent dimensions
                ct_param_grp = ct_grp.create_group('parameters')
                essential_params = ['mass', 'true_mass_Mjup', 'T_int', 'T_int_input_dial', 
                                    'T_irr', 'Met', 'core_mass_earth', 'f_sed', 'kzz']
                for k in essential_params:
                    if k in params:
                        ct_param_grp.attrs[k] = params[k]
                
                # Cooltrack: Extract only radius, cooling rate, and entropy curve
                ct_int_grp = ct_grp.create_group('interior_raw')
                if int_raw:
                    for attr_key in ['R_total', 'dt_ds_total']:
                        if attr_key in int_raw:
                            ct_int_grp.attrs[attr_key] = int_raw[attr_key]
                    if 'S' in int_raw:
                        ct_int_grp.create_dataset('S', data=np.asarray(int_raw['S']), compression="gzip")
                
                # Cooltrack: Strip out raw spectra, keep only the target band fluxes
                if phot_data and 'bands' in phot_data:
                    ct_phot_grp = ct_grp.create_group('photometry')
                    ct_bands_grp = ct_phot_grp.create_group('bands')
                    for filt_id, metrics in phot_data['bands'].items():
                        safe_filt_id = filt_id.replace('/', '_')
                        f_grp = ct_bands_grp.create_group(safe_filt_id)
                        if 'flux_W_m2_um' in metrics:
                            f_grp.attrs['flux_W_m2_um'] = metrics['flux_W_m2_um']
                # =================================================================

            summary_catalog.append(catalog_entry)
            
            if (idx + 1) % 50 == 0:
                logging.info(f"Processed {idx + 1}/{total_files} files...")

    # --- 6. Save the Catalog ---
    df_catalog = pd.DataFrame(summary_catalog)
    
    # Reorder columns to keep the basics at the front
    cols = df_catalog.columns.tolist()
    if 'R_1bar_Rjup' in cols:
        cols.insert(3, cols.pop(cols.index('R_1bar_Rjup')))
    df_catalog = df_catalog[cols]
    
    df_catalog.to_csv(csv_path, index=False)
    
    logging.info(f"✅ Grid Compilation Complete!")
    logging.info(f"📊 Catalog saved to: {csv_path}")
    logging.info(f"🗄️  Master Data stored in:  {h5_path}")
    logging.info(f"⚡ CoolTrack Data stored in: {cooltrack_path}")

if __name__ == "__main__":
    TARGET_GRID_DIR = "../outputs/grid_run"
    OUTPUT_PREFIX = "../outputs/master_grid"
    compile_exoweave_grid(TARGET_GRID_DIR, OUTPUT_PREFIX)