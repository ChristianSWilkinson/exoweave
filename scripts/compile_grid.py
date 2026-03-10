import os
import pickle
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# Set up clean logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
                else:
                    status = 'max_iterations_reached'

            # --- 2. Build the Base Catalog Entry ---
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
                'iterations': data.get('iterations', np.nan),
                'P_link_bar': params.get('p_link_bar', np.nan),
                'R_total_m': np.nan,      
                'R_1bar_Rjup': np.nan,    
                'original_file': pkl_file.name
            }

            # --- 3. Extract Arrays, Interpolate Radius, and Pull Photometry ---
            prof_df = data.get('profile') if 'profile' in data else data.get('stitched_profile')
            atm_df = data.get('atmosphere_raw')
            int_raw = data.get('interior_raw', {})
            phot_data = data.get('photometry', {})
            
            if status in ['target_reached', 'max_iterations_reached'] and prof_df is not None and not prof_df.empty:
                r_cols = [c for c in prof_df.columns if 'rad' in c.lower()]
                p_cols = [c for c in prof_df.columns if 'press' in c.lower()]
                
                if r_cols:
                    r_col = r_cols[0]
                    catalog_entry['R_total_m'] = prof_df[r_col].max()
                    
                    if p_cols:
                        p_col = p_cols[0]
                        log_p = np.log10(prof_df[p_col].values)
                        r_m = prof_df[r_col].values
                        sort_idx = np.argsort(log_p)
                        r_1bar_m = np.interp(0.0, log_p[sort_idx], r_m[sort_idx])
                        catalog_entry['R_1bar_Rjup'] = r_1bar_m / R_JUPITER_M

                if 'bands' in phot_data:
                    for filter_id, metrics in phot_data['bands'].items():
                        safe_name = filter_id.replace('/', '_')
                        catalog_entry[f"{safe_name}_flux_Wm2um"] = metrics.get('flux_W_m2_um', np.nan)
                        catalog_entry[f"{safe_name}_flux_Jy"] = metrics.get('flux_Jy', np.nan)

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