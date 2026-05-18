import os
import pickle
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import json

# Set up clean logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
            
        p_link_bar = catalog_entry.get('P_link_bar', 1000.0)
        if np.isnan(p_link_bar): 
            p_link_bar = 1000.0
            
        max_valid_pressure_pa = p_link_bar * 1e5 * 1000.0
        mask = prof_df['Pressure_Pa'].values < max_valid_pressure_pa
        
        t_values = prof_df.loc[mask, 'Temperature_K'].values
        p_values = prof_df.loc[mask, 'Pressure_Pa'].values
        
        if len(t_values) < 2:
            return True, "OK" 
            
        dt = np.abs(np.diff(t_values))
        dp_log = np.abs(np.diff(np.log10(p_values)))
        dp_log = np.maximum(dp_log, 1e-8) 
        
        gradients = dt / dp_log
        
        max_idx = np.argmax(gradients)
        worst_grad = gradients[max_idx]
        worst_jump = dt[max_idx]
        median_jump = np.median(dt)
        
        if worst_jump > min_jump_k and worst_jump > (5 * median_jump) and worst_grad > grad_threshold:
            return False, f"PT_DISCONT_{worst_jump:.0f}K_grad{worst_grad:.1e}"
            
        return True, "OK" 
    
    @staticmethod
    def check_pressure_resolution(prof_df, max_dp_log=1):
        """Ensures the profile does not have massive gaps in the pressure grid."""
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Pressure_Pa' not in prof_df.columns:
            return False, "MISSING_PRESSURE_COLUMN"
            
        p_values = prof_df['Pressure_Pa'].values
        dp_log = np.abs(np.diff(np.log10(p_values)))
        max_step = np.max(dp_log)
        
        if max_step > max_dp_log:
            return False, f"GRID_GAP_{max_step:.2f}dex"
            
        return True, "OK"

    @staticmethod
    def check_physical_radius(r_total_m):
        """Flags planets with highly unphysical total radii."""
        if r_total_m is None or np.isnan(r_total_m):
            return False, "NO_RADIUS_DATA"
            
        r_jup = r_total_m / 71492000.0
        if r_jup < 0.3 or r_jup > 5.0:
            return False, f"UNPHYSICAL_RADIUS_{r_jup:.2f}Rj"
            
        return True, "OK"

    @staticmethod
    def check_cooling_rate(int_raw):
        """Checks interior_raw for a valid dt_ds_total."""
        if not int_raw:
            return False, "NO_INTERIOR_DATA"
            
        dt_ds = int_raw.get('dt_ds_total', 0.0)
        
        if dt_ds == 0.0 or np.isnan(dt_ds):
            return False, "FLAT_COOLING"
        if dt_ds >= 0.0 or np.isnan(dt_ds):
            return False, "POSITIVE_COOLING"
            
        return True, "OK"
    
    @staticmethod
    def check_pt_percentage_jump(prof_df, threshold_pct=50.0):
        """Flags profiles where the temperature jumps by an extreme percentage."""
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Temperature_K' not in prof_df.columns:
            return False, "MISSING_TEMP_COLUMN"
            
        t_values = prof_df['Temperature_K'].values
        if len(t_values) < 2:
            return True, "OK"
            
        dt = np.abs(np.diff(t_values))
        t_prev = np.maximum(t_values[:-1], 1e-8)
        dt_percent = (dt / t_prev) * 100.0
        
        max_idx = np.argmax(dt_percent)
        max_pct_val = dt_percent[max_idx]
        
        if max_pct_val > threshold_pct:
            return False, f"PT_JUMP_{max_pct_val:.0f}PCT"
            
        return True, "OK"

    @staticmethod
    def check_max_tint(t_int_true_k, max_t_int=2000.0):
        """Flags models where the true internal temperature exceeds validity."""
        if t_int_true_k is None or np.isnan(t_int_true_k):
            return False, "NO_TINT_DATA"
            
        if t_int_true_k > max_t_int:
            return False, f"TINT_TOO_HIGH_{t_int_true_k:.0f}K"
            
        return True, "OK"

    @staticmethod
    def check_negative_t_drop(prof_df, min_p_bar=0.01, max_p_bar=100.0, window=3, max_drop_k=50.0):
        """
        Flags unphysical temperature inversions where temperature plummets 
        as pressure increases (going deeper). Checks over a rolling layer window
        within a specific pressure zone (default: 0.1 to 100 bars).
        """
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Temperature_K' not in prof_df.columns or 'Pressure_Pa' not in prof_df.columns:
            return False, "MISSING_COLUMNS"
            
        # Convert to bars and filter for the targeted pressure region
        p_bars = prof_df['Pressure_Pa'].values / 1e5
        
        # --- THE FIX: Filter between min_p_bar and max_p_bar ---
        mask = (p_bars >= min_p_bar) & (p_bars <= max_p_bar)
        
        p_zone = p_bars[mask]
        t_zone = prof_df.loc[mask, 'Temperature_K'].values
        
        # Sort top-down (lowest pressure to highest pressure)
        sort_idx = np.argsort(p_zone)
        t_sorted = t_zone[sort_idx]
        
        # If the atmosphere is too thin to check the window, pass it
        if len(t_sorted) < window + 1:
            return True, "OK"
            
        # Scan through the array with our rolling window
        for i in range(len(t_sorted) - window):
            t_shallow = t_sorted[i]
            t_deep = t_sorted[i + window]
            
            # If the deep layer is significantly colder than the shallow layer
            if (t_shallow - t_deep) > max_drop_k:
                return False, f"UNPHYSICAL_T_DROP_{int(t_shallow - t_deep)}K"
                
        return True, "OK"
    
    @staticmethod
    def check_linkage_continuity(prof_df, catalog_entry, max_jump_k=100.0):
        """
        Detects unphysical temperature discontinuities specifically at the 
        atmosphere-interior linkage boundary.
        """
        if prof_df is None or prof_df.empty: 
            return False, "NO_PROFILE_DATA"
            
        if 'Temperature_K' not in prof_df.columns or 'Pressure_bar' not in prof_df.columns:
            return False, "MISSING_COLUMNS"
            
        p_link_bar = catalog_entry.get('P_link_bar')
        if p_link_bar is None or np.isnan(p_link_bar): 
            return True, "OK" # Skip if we don't know where the boundary is
            
        p_bars = prof_df['Pressure_bar'].values
        t_values = prof_df['Temperature_K'].values
        
        # Ensure the profile is sorted top-down
        sort_idx = np.argsort(p_bars)
        p_sorted = p_bars[sort_idx]
        t_sorted = t_values[sort_idx]
        
        # Find the exact index where the atmosphere transitions to the interior
        diffs = np.abs(p_sorted - p_link_bar)
        link_idx = np.argmin(diffs)
        
        # Check a narrow window (e.g., +/- 2 layers) around the linkage point
        start_idx = max(0, link_idx - 2)
        end_idx = min(len(t_sorted) - 1, link_idx + 2)
        
        for i in range(start_idx, end_idx):
            t_jump = np.abs(t_sorted[i+1] - t_sorted[i])
            p_ratio = p_sorted[i+1] / p_sorted[i]
            
            # If the pressure step is small (< 1.5x) but the temperature jumps violently,
            # it means the two models failed to stitch smoothly.
            if p_ratio < 1.5 and t_jump > max_jump_k:
                return False, f"LINK_DISCONT_{t_jump:.0f}K_at_{p_sorted[i]:.1f}bar"
                
        return True, "OK"

    @staticmethod
    def validate(data, catalog_entry):
        """Main validation router."""
        if data.get('status') == 'failed' or 'failure_reason' in data:
            return False, "SOLVER_FAILED"

        prof_df = data.get('stitched_profile') if 'stitched_profile' in data else data.get('profile')
        int_raw = data.get('interior_raw', {})
        r_total_m = int_raw.get('R_total', catalog_entry.get('R_total_m'))
        t_int_true = catalog_entry.get('T_int_true_K')

        # 1. Grid Resolution Check
        grid_ok, grid_reason = ExoFilter.check_pressure_resolution(prof_df)
        if not grid_ok: return False, grid_reason

        # 2. Relative Percentage Jump Check
        pct_ok, pct_reason = ExoFilter.check_pt_percentage_jump(prof_df, threshold_pct=50.0)
        if not pct_ok: return False, pct_reason
        
        # 3. Physical Radius Check
        rad_ok, rad_reason = ExoFilter.check_physical_radius(r_total_m)
        if not rad_ok: return False, rad_reason
        
        # 4. Maximum Valid Temperature Check
        tint_ok, tint_reason = ExoFilter.check_max_tint(t_int_true, max_t_int=2000.0)
        if not tint_ok: return False, tint_reason
        
        # 5. Cooling Rate Check
        cool_ok, cool_reason = ExoFilter.check_cooling_rate(int_raw)
        if not cool_ok: return False, cool_reason
        
        # 6. Negative Temperature Drop (Inversion) Check
        # Tuned to > 50 K drop over 3 layers at P < 100 bars
        drop_ok, drop_reason = ExoFilter.check_negative_t_drop(prof_df, min_p_bar=0.1, max_p_bar=100.0, window=3, max_drop_k=100.0)
        if not drop_ok: return False, drop_reason

        link_ok, link_reason = ExoFilter.check_linkage_continuity(prof_df, catalog_entry, max_jump_k=200.0)
        if not link_ok: return False, link_reason
        
        return True, "VALID"
        
# =============================================================================
# RECURSIVE HDF5 SAVER 
# =============================================================================
def _recursively_save_dict_to_hdf5(h5_obj, d):
    """
    Recursively saves a Python dictionary into an HDF5 group.
    Handles nested dicts, pandas DataFrames, numpy arrays, lists, and scalars.
    """
    for key, item in d.items():
        safe_key = str(key).replace('/', '_')
        
        if isinstance(item, dict):
            sub_group = h5_obj.create_group(safe_key)
            _recursively_save_dict_to_hdf5(sub_group, item)
            
        elif isinstance(item, pd.DataFrame):
            df_group = h5_obj.create_group(safe_key)
            for col in item.columns:
                safe_col = str(col).replace('/', '_')
                col_data = item[col].values
                
                # 🚨 THE FIX: Unpack hidden arrays from 1-row DataFrames (ExoREM format)
                if col_data.dtype == 'O':
                    if len(col_data) == 1 and isinstance(col_data[0], (list, np.ndarray)):
                        col_data = np.array(col_data[0])
                
                # If it's still an object/string array after unpacking, safely stringify
                if col_data.dtype == 'O' or str(col_data.dtype).startswith('<U'):
                    try:
                        col_data = np.array([str(val) for val in col_data], dtype=h5py.string_dtype(encoding='utf-8'))
                    except Exception:
                        pass
                        
                try:
                    df_group.create_dataset(safe_col, data=col_data)
                except Exception:
                    df_group.attrs[safe_col] = str(col_data)
                    
        elif isinstance(item, (np.ndarray, list, tuple)):
            try:
                arr = np.array(item)
                
                # Unpack nested lists if necessary
                if arr.dtype == 'O' and len(arr) == 1 and isinstance(arr[0], (list, np.ndarray)):
                    arr = np.array(arr[0])
                    
                if arr.dtype == 'O' or str(arr.dtype).startswith('<U'):
                    arr = np.array([str(val) for val in arr], dtype=h5py.string_dtype(encoding='utf-8'))
                h5_obj.create_dataset(safe_key, data=arr)
            except Exception:
                h5_obj.attrs[safe_key] = str(item)
                
        elif isinstance(item, (int, float, str, bytes, bool, np.generic)):
            h5_obj.attrs[safe_key] = item
            
        elif item is None:
            h5_obj.attrs[safe_key] = "None"
            
        else:
            h5_obj.attrs[safe_key] = str(item)


    @staticmethod
    def cross_model_monotonicity_check(df_catalog):
        """
        Pass 2 Filter: Detects artificially inflated cold models in a sparse LHS grid.
        """
        logging.info("🕵️ Running Cross-Model Neighborhood Monotonicity Filter...")
        
        valid_mask = df_catalog['qc_status'] == 'VALID'
        valid_df = df_catalog[valid_mask].copy()
        anomalous_indices = []
        
        for idx, rowA in valid_df.iterrows():
            mass_match = np.abs(valid_df['target_mass_Mjup'] - rowA['target_mass_Mjup']) / rowA['target_mass_Mjup'] < 0.05
            tirr_match = np.abs(valid_df['T_irr_K'] - rowA['T_irr_K']) < 50.0
            met_match = np.abs(valid_df['metallicity'] - rowA['metallicity']) < 0.2
            
            neighbors = valid_df[mass_match & tirr_match & met_match]
            
            for _, rowB in neighbors.iterrows():
                if rowA['model_id'] == rowB['model_id']: continue
                
                dT_int = rowB['T_int_dial_K'] - rowA['T_int_dial_K']
                dRadius = rowB['R_total_m'] - rowA['R_total_m']
                
                if dT_int > 150.0 and dRadius < -(0.05 * 71492000.0):
                    anomalous_indices.append(idx)
                    break
                    
        df_catalog.loc[anomalous_indices, 'qc_status'] = "NON_MONOTONIC_RADIUS_ANOMALY"
        logging.info(f"🚩 Flagged {len(anomalous_indices)} models for non-monotonic junction failures.")
        return df_catalog
    

# =============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# =============================================================================

def _extract_metadata_worker(args):
    """Worker function for Pass 1: Loads pickle and extracts exact custom metadata/QC."""
    idx, pkl_file = args
    model_id = f"model_{idx:05d}"
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return {'model_id': model_id, 'qc_status': f"FILE_READ_ERROR: {e}", 'pkl_path': pkl_file}

    # --- YOUR CUSTOM STATUS PARSING ---
    if 'failed' in Path(pkl_file).parts:
        status = 'crashed'
    else:
        raw_status = data.get('status', 'converged')
        if raw_status == 'converged':
            status = 'target_reached'
        elif raw_status == 'intermediate':
            status = 'intermediate_step'
        else:
            status = 'max_iterations_reached'

    params = data.get('final_params', data.get('parameters', {}))
    iters = data.get('iterations', data.get('iteration', np.nan))

    # --- YOUR EXACT CATALOG ENTRY ---
    catalog_entry = {
        'model_id': model_id,
        'status': status,
        'target_mass_Mjup': params.get('mass', np.nan),
        'true_mass_Mjup': params.get('true_mass_Mjup', np.nan),
        'T_int_dial_K': params.get('T_int_input_dial', params.get('T_int', np.nan)),
        'T_int_true_K': params.get('T_int', np.nan),
        'T_irr_K': params.get('T_irr', np.nan),
        'T_eff_K': params.get('T_eff', np.nan),
        'metallicity': params.get('Met', np.nan),
        'core_mass_Me': params.get('core_mass_earth', np.nan),
        'f_sed': params.get('f_sed', np.nan),
        'kzz': params.get('kzz', np.nan),
        'iterations': iters, 
        'P_link_bar': params.get('p_link_bar', np.nan),
        'S_physical': np.nan, 
        'dsdt_J_K_kg_s': np.nan,
        'R_total_m': np.nan,      
        'R_1bar_Rjup': np.nan,    
        'original_file': Path(pkl_file).name,
        'pkl_path': pkl_file  # Required for parallel routing
    }

    prof_df = data.get('profile') if 'profile' in data else data.get('stitched_profile')
    int_raw = data.get('interior_raw', {})
    cool_raw = data.get('cooling_metrics', {})
    phot_data = data.get('photometry', {})

    # --- YOUR CUSTOM R_1bar AND R_total EXTRACTION ---
    if prof_df is not None and 'Pressure_bar' in prof_df.columns and 'Radius_m' in prof_df.columns:
        try:
            sorted_prof = prof_df.sort_values('Pressure_bar')
            r_1bar_m = np.interp(1.0, sorted_prof['Pressure_bar'], sorted_prof['Radius_m'])
            catalog_entry['R_1bar_Rjup'] = r_1bar_m / 71492000.0
            if int_raw and 'R_total' in int_raw:
                catalog_entry['R_total_m'] = int_raw['R_total']
        except Exception:
            pass

    # --- YOUR CUSTOM PHOTOMETRY FLATTENING ---
    if phot_data and 'bands' in phot_data:
        for filt_id, metrics in phot_data['bands'].items():
            safe_filt_id = filt_id.replace('/', '_')
            if 'flux_W_m2_um' in metrics:
                catalog_entry[f"{safe_filt_id}_flux_Wm2um"] = metrics['flux_W_m2_um']
            if 'flux_Jy' in metrics:
                catalog_entry[f"{safe_filt_id}_flux_Jy"] = metrics['flux_Jy']

    # --- YOUR CUSTOM ENTROPY & COOLING EXTRACTION ---
    if int_raw and 'S' in int_raw:
        try:
            s_val = int_raw['S']
            catalog_entry['S_physical'] = float(np.max(np.asarray(s_val).flatten())) if isinstance(s_val, (list, np.ndarray)) else float(s_val)
        except Exception:
            logging.debug(f"⚠️ Could not extract S_physical for {model_id}")
            pass
            
    if cool_raw and 'ds_dt' in cool_raw:
        try:
            catalog_entry['dsdt_J_K_kg_s'] = float(cool_raw['ds_dt'])
        except Exception:
            pass

    # --- QUALITY CONTROL ---
    if hasattr(ExoFilter, 'validate'):
        is_valid, reason = ExoFilter.validate(data, catalog_entry)
        catalog_entry['qc_status'] = reason if not is_valid else "VALID"
    else:
        catalog_entry['qc_status'] = "VALID"

    return catalog_entry


def _load_heavy_data_worker(args):
    """Worker function for Pass 3: Heavily unpickles arrays to hand to main thread."""
    model_id, pkl_file = args
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        return model_id, pkl_file, data, None
    except Exception as e:
        return model_id, pkl_file, None, str(e)
    

# =============================================================================
# MAIN COMPILER FUNCTION
# =============================================================================

def compile_exoweave_grid(input_dir: str, output_prefix: str, delete_failed_models: bool = False):
    in_path = Path(input_dir)
    if not in_path.exists():
        logging.error(f"❌ Input directory not found: {input_dir}")
        return

    pkl_files = list(in_path.glob("**/*.pkl"))
    pkl_files.sort()
    total_files = len(pkl_files)
    
    if total_files == 0:
        logging.error("❌ No .pkl files found to compile.")
        return
        
    max_cores = max(1, os.cpu_count() - 1)
    
    # -------------------------------------------------------------------------
    # PASS 1: METADATA EXTRACTION (PARALLEL)
    # -------------------------------------------------------------------------
    summary_catalog = []
    logging.info(f"🚀 PASS 1: Extracting metadata for {total_files} models using {max_cores} CPU cores...")
    
    tasks = [(idx, pkl_file) for idx, pkl_file in enumerate(pkl_files)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        for i, catalog_entry in enumerate(executor.map(_extract_metadata_worker, tasks), 1):
            summary_catalog.append(catalog_entry)
            if i % 250 == 0:
                logging.info(f"   Pass 1 Progress: {i}/{total_files} files processed...")

    df_catalog = pd.DataFrame(summary_catalog)
    
    # -------------------------------------------------------------------------
    # PASS 2: CROSS-MODEL MONOTONICITY FILTER
    # -------------------------------------------------------------------------
    if hasattr(ExoFilter, 'cross_model_monotonicity_check'):
        df_catalog = ExoFilter.cross_model_monotonicity_check(df_catalog)

    # -------------------------------------------------------------------------
    # PASS 3: HDF5 BINARY WRITING (PARALLEL READ, SEQUENTIAL WRITE)
    # -------------------------------------------------------------------------
    valid_rows = df_catalog[df_catalog['qc_status'] == 'VALID']
    failed_rows = df_catalog[df_catalog['qc_status'] != 'VALID']
    
    logging.info(f"💾 PASS 3: Writing {len(valid_rows)} Valid Models to HDF5 Binaries...")
    
    h5_path = Path(f"{output_prefix}_data.h5")
    cooltrack_path = Path(f"{output_prefix}_cooltrack.h5")
    csv_path = Path(f"{output_prefix}_catalog.csv")
    
    # Permanently delete failed models if requested
    if delete_failed_models:
        for _, row in failed_rows.iterrows():
            try: 
                row['pkl_path'].unlink()
                logging.debug(f"🗑️ Deleted poorly converged file: {row['pkl_path'].name}")
            except Exception: 
                pass

    pass3_tasks = [(row['model_id'], row['pkl_path']) for _, row in valid_rows.iterrows()]
    
    with h5py.File(h5_path, 'w') as h5_master, h5py.File(cooltrack_path, 'w') as h5_cool:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            future_to_model = {executor.submit(_load_heavy_data_worker, task): task for task in pass3_tasks}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_model):
                model_id, pkl_file, data, err = future.result()
                completed_count += 1
                
                if err: 
                    logging.error(f"❌ Failed to load {pkl_file.name} for Pass 3: {err}")
                    continue

                # =============================================================
                # YOUR EXACT HDF5 EXPORT LOGIC 
                # =============================================================
                params = data.get('final_params', data.get('parameters', {}))
                int_raw = data.get('interior_raw', {})
                cool_raw = data.get('cooling_metrics', {})
                phot_data = data.get('photometry', {})

                # 1. Master Export (Assumes _recursively_save_dict_to_hdf5 is defined globally in your script)
                model_grp = h5_master.create_group(model_id)
                _recursively_save_dict_to_hdf5(model_grp, data)

                # 2. Cooltrack Export
                ct_grp = h5_cool.create_group(model_id)
                ct_param = ct_grp.create_group('parameters')
                for key in ['mass', 'true_mass_Mjup', 'T_int', 'T_int_input_dial', 'T_irr', 'Met', 'core_mass_earth', 'f_sed', 'kzz', 'T_eff']:
                    val = params.get(key, np.nan)
                    
                    if isinstance(val, (dict, list, tuple)):
                        ct_param.attrs[key] = json.dumps(val)
                    elif val is None:
                        ct_param.attrs[key] = "None"
                    else:
                        ct_param.attrs[key] = val
                            
                if int_raw:
                    ct_int = ct_grp.create_group('interior_raw')
                    for key in ['dt_ds_total', 'M_total', 'R_total', 'S']:
                        if key in int_raw:
                            val = int_raw[key]
                            if isinstance(val, (np.ndarray, list)):
                                ct_int.create_dataset(key, data=val, compression="gzip")
                            else:
                                ct_int.attrs[key] = val

                if cool_raw:
                    ct_cooling_metrics = ct_grp.create_group('cooling_metrics')
                    for key in ['L_int', 'ds_dt', 'dt_ds']:
                        if key in cool_raw:
                            ct_cooling_metrics.attrs[key] = cool_raw[key]

                if phot_data and 'bands' in phot_data:
                    ct_phot = ct_grp.create_group('photometry')
                    ct_bands = ct_phot.create_group('bands')
                    for filt_id, metrics in phot_data['bands'].items():
                        safe_filt_id = filt_id.replace('/', '_')
                        f_grp = ct_bands.create_group(safe_filt_id)
                        if 'flux_W_m2_um' in metrics:
                            f_grp.attrs['flux_W_m2_um'] = metrics['flux_W_m2_um']

                # =============================================================
                
                if completed_count % 100 == 0:
                    logging.info(f"   Pass 3 Progress: {completed_count}/{len(pass3_tasks)} binary exports written...")

    # -------------------------------------------------------------------------
    # 6. SAVE CSV CATALOG
    # -------------------------------------------------------------------------
    df_catalog = df_catalog.drop(columns=['pkl_path'])
    
    cols = df_catalog.columns.tolist()
    if 'R_1bar_Rjup' in cols:
        cols.insert(3, cols.pop(cols.index('R_1bar_Rjup')))
    df_catalog = df_catalog[cols]
    
    df_catalog.to_csv(csv_path, index=False)
    
    logging.info(f"✅ Grid Compilation Complete!")
    logging.info(f"📊 Catalog saved to: {csv_path}")
    logging.info(f"🗄️ Master Data stored in: {h5_path}")
    logging.info(f"🧊 CoolTrack Extract stored in: {cooltrack_path}")

# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    TARGET_GRID_DIR = "../outputs/grid_run_5clouds"
    OUTPUT_PREFIX = "../outputs/master_grid"

    compile_exoweave_grid(
        input_dir=TARGET_GRID_DIR, 
        output_prefix=OUTPUT_PREFIX, 
        delete_failed_models=False
    )