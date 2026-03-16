import os
import pickle
import logging
import numpy as np
from pathlib import Path

# Import your SVO fetcher
from exowrap.photometry import get_svo_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_FILTER_CACHE = {}

def extract_filter_list_from_pkls(pkl_dir: Path) -> list:
    """Scans all .pkl files to build a master list of filters used."""
    logging.info(f"🔍 Scanning {pkl_dir} to identify all used filters...")
    master_filters = set()
    
    for pkl_file in pkl_dir.glob("**/*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
            if 'photometry' in data and 'bands' in data['photometry']:
                # In the .pkl files, the keys are usually the raw SVO strings (e.g., 'JWST/NIRCam.F444W')
                for filter_id in data['photometry']['bands'].keys():
                    master_filters.add(filter_id)
        except Exception:
            continue
            
    filter_list = sorted(list(master_filters))
    logging.info(f"📋 Found {len(filter_list)} unique filters across the .pkl files.")
    return filter_list

def patch_pkl_photometry(target_dir: str, target_filters: list = None, force_recalc: bool = False):
    """
    Reads .pkl files, calculates missing or corrupted photometry, 
    and safely overwrites the original .pkl file.
    """
    pkl_dir = Path(target_dir)
    if not pkl_dir.exists():
        logging.error(f"❌ Directory {target_dir} not found.")
        return

    # --- AUTO-DETECT FILTERS IF LIST IS EMPTY ---
    if not target_filters:
        target_filters = extract_filter_list_from_pkls(pkl_dir)
        
    if not target_filters:
        logging.error("❌ No filters provided and none found in the .pkl files.")
        return
        
    logging.info(f"🎯 Target List: {len(target_filters)} filters queued for processing.")
    
    pkl_files = list(pkl_dir.glob("**/*.pkl"))
    models_updated = 0

    for idx, pkl_file in enumerate(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logging.warning(f"⚠️ Could not read {pkl_file.name}: {e}")
            continue
            
        # 1. Skip if it's a failed model or missing raw spectra
        if data.get('status') not in ['converged', 'target_reached', 'max_iterations_reached']:
            continue
            
        if 'photometry' not in data:
            continue
            
        phot_data = data['photometry']
        
        # Check if arrays are present and valid
        exo_wl = phot_data.get('wavelength_um')
        exo_flux = phot_data.get('emission_flux_W_m2_um')
        
        if exo_wl is None or exo_flux is None:
            continue
            
        valid = exo_wl > 0
        exo_wl = exo_wl[valid]
        exo_flux = exo_flux[valid]

        sort_idx = np.argsort(exo_wl)
        exo_wl = exo_wl[sort_idx]
        exo_flux = exo_flux[sort_idx]
        
        if 'bands' not in phot_data:
            phot_data['bands'] = {}
        bands_dict = phot_data['bands']
        
        updated_this_model = False
        
        for filter_id in target_filters:
            # Skip if we already have it and aren't forcing a recalculation
            if not force_recalc and filter_id in bands_dict:
                continue
                
            try:
                # 2. Fetch Filter
                if filter_id not in _FILTER_CACHE:
                    logging.info(f"🌐 Downloading {filter_id} from SVO...")
                    _FILTER_CACHE[filter_id] = get_svo_filter(filter_id)
                    
                filt_wav, filt_trans = _FILTER_CACHE[filter_id]
                
                # 3. Integrate
                interp_trans = np.interp(exo_wl, filt_wav, filt_trans, left=0.0, right=0.0)
                if np.sum(interp_trans) == 0:
                    continue 

                eff_wav = np.trapz(interp_trans * exo_wl, exo_wl) / np.trapz(interp_trans, exo_wl)
                numerator = np.trapz(exo_flux * interp_trans * exo_wl, exo_wl)
                denominator = np.trapz(interp_trans * exo_wl, exo_wl)

                phot_flux_flambda = numerator / denominator

                c_um_s = 299792458.0 * 1e6
                phot_flux_fnu = phot_flux_flambda * (eff_wav**2) / c_um_s
                phot_flux_jy = phot_flux_fnu * 1e26
                
                # 4. Update the dictionary in memory
                bands_dict[filter_id] = {
                    "filter_id": filter_id,
                    "effective_wavelength_um": eff_wav,
                    "flux_W_m2_um": phot_flux_flambda,
                    "flux_Jy": phot_flux_jy
                }
                
                updated_this_model = True
                
            except Exception as e:
                pass # Silently skip filters that fail to download or integrate
                
        # 5. Safely save back to the .pkl file if changes were made
        if updated_this_model:
            temp_file = pkl_file.with_suffix('.pkl.tmp')
            try:
                # Write to a temporary file first to prevent corruption if the script crashes
                with open(temp_file, 'wb') as f:
                    pickle.dump(data, f)
                # Replace the old file with the new one
                temp_file.replace(pkl_file)
                models_updated += 1
            except Exception as e:
                logging.error(f"❌ Failed to save updates to {pkl_file.name}: {e}")
                if temp_file.exists():
                    temp_file.unlink() # Clean up temp file
            
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1}/{len(pkl_files)} models...")

    logging.info(f"✅ Successfully patched {models_updated} .pkl files!")
    logging.info("🚀 You can now run `compile_grid.py` to rebuild your HDF5 and CSV.")

if __name__ == "__main__":
    
    # Point this to your main output directory containing the target/ and failed/ folders
    GRID_OUTPUT_DIR = "../outputs/grid_run"
    
    # Leave empty to auto-detect from the .pkls
    FILTERS_TO_FIX = [] 
    
    # Set to True to mathematically overwrite existing photometric data
    FORCE_RECALCULATION = True 
    
    patch_pkl_photometry(
        target_dir=GRID_OUTPUT_DIR, 
        target_filters=FILTERS_TO_FIX, 
        force_recalc=FORCE_RECALCULATION 
    )