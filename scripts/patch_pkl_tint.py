import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================================
# 1. The Core Physics Logic (Adapted for direct DataFrame access)
# =====================================================================
def calculate_rcb_tint_from_df(df: pd.DataFrame, pressure_threshold_bar: float = 10.0) -> float:
    """
    Applies the direction-agnostic RCB Sniper logic directly to the raw 
    ExoREM Pandas DataFrame extracted from the .pkl file.
    """
    try:
        # Extract arrays (assuming they are stored in the first row of the dataframe)
        p_bar = np.asarray(df['/outputs/levels/pressure'].iloc[0]) / 1e5
        rad_int = np.asarray(df['/outputs/levels/radiosity_internal'].iloc[0])
        is_conv = np.asarray(df['/outputs/levels/is_convective'].iloc[0])
        
        n_layers = len(is_conv)
        clear_space = 3
        p_rcb = np.nan
        
        # Check if atmosphere is already fully radiative deep down
        p_mask = p_bar >= pressure_threshold_bar
        if np.any(p_mask) and np.all(is_conv[p_mask] == 0):
            p_rcb = np.max(p_bar[p_mask])
        else:
            bottom_idx = np.argmax(p_bar)
            step = 1 if bottom_idx == 0 else -1
            idx = bottom_idx
            
            def is_valid(i): return 0 <= i < n_layers
            
            while is_valid(idx):
                while is_valid(idx) and is_conv[idx] == 0: idx += step
                if not is_valid(idx): break
                
                while is_valid(idx) and is_conv[idx] == 1: idx += step
                if not is_valid(idx): break
                    
                if step == 1:
                    block = is_conv[idx : min(idx + clear_space, n_layers)]
                else:
                    block = is_conv[max(0, idx - clear_space + 1) : idx + 1]
                    
                if np.all(block == 0):
                    p_rcb = p_bar[idx - step]
                    break
                    
        # Extract the flux at the true RCB
        if not np.isnan(p_rcb):
            idx_rcb = np.argmin(np.abs(p_bar - p_rcb))
            bottom_idx = np.argmax(p_bar)
            step = 1 if bottom_idx == 0 else -1
            
            if step == 1:
                slice_indices = list(range(idx_rcb, min(idx_rcb + 3, n_layers)))
            else:
                slice_indices = list(range(max(0, idx_rcb - 2), idx_rcb + 1))
            
            rcb_flux = np.nanmean(rad_int[slice_indices])
            
            if rcb_flux > 0:
                sigma_sb = 5.670374419e-8
                return float((rcb_flux / sigma_sb) ** 0.25)
                
    except Exception as e:
        pass
        
    return np.nan

# =====================================================================
# 2. The Batch Update Script
# =====================================================================
def update_pkl_grid(grid_dir_path: str):
    grid_dir = Path(grid_dir_path)
    pkl_files = list(grid_dir.glob("**/*.pkl"))
    
    if not pkl_files:
        print("❌ No .pkl files found in that directory!")
        return

    print(f"🚀 Found {len(pkl_files)} .pkl files. Starting batch correction...\n")
    
    updated_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, filepath in enumerate(pkl_files):
        try:
            # 1. Load the data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check if it has the required data to perform the operation
            if 'atmosphere_raw' not in data or 'parameters' not in data:
                skipped_count += 1
                continue
            
            df_atm = data['atmosphere_raw']
            
            # 2. Calculate the new T_int
            new_t_int = calculate_rcb_tint_from_df(df_atm)
            
            if not np.isnan(new_t_int):
                old_t_int = data['parameters'].get('T_int', np.nan)
                
                # 3. Update the dictionary in memory
                data['parameters']['T_int'] = new_t_int
                
                # 4. Write back to the exact same file (overwriting it)
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                
                updated_count += 1
                
                # Print progress every 100 files to avoid flooding terminal
                if updated_count % 100 == 0:
                    print(f"  -> Processed {idx + 1}/{len(pkl_files)}... (Last correction: {old_t_int:.1f}K -> {new_t_int:.1f}K)")
            else:
                # The script couldn't find a valid flux (e.g., negative flux / non-converged)
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            print(f"⚠️ Error processing file {filepath.name}: {e}")

    print("\n" + "="*50)
    print("✅ BATCH UPDATE COMPLETE")
    print("="*50)
    print(f"Total Files Scanned : {len(pkl_files)}")
    print(f"Successfully Updated: {updated_count}")
    print(f"Skipped (No Data)   : {skipped_count}")
    print(f"Failed (Bad/Neg Flux): {failed_count}")

# Execute the script
update_pkl_grid("../outputs/grid_run")