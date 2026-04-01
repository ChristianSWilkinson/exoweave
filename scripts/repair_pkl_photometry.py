import pickle
import logging
from pathlib import Path

# Import your existing ExoWeave architecture
from exowrap.output import ExoremOut
from exoweave.physics import calculate_comprehensive_photometry

logging.basicConfig(level=logging.INFO, format="%(message)s")

def repair_photometry_in_pkls(target_dir: str):
    """
    Scans for .pkl files missing 'photometry' in both 'target' and 'steps' 
    directories, reconstructs the ExoREM output from 'atmosphere_raw', 
    calculates the missing photometry, and occasionally prints sanity checks.
    """
    pkl_dir = Path(target_dir)
    if not pkl_dir.exists():
        logging.error(f"❌ Directory {target_dir} not found.")
        return

    # Grab .pkl files from BOTH the target and steps directories
    target_files = list(pkl_dir.glob("target/*.pkl"))
    step_files = list(pkl_dir.glob("steps/*.pkl"))
    pkl_files = target_files + step_files
    
    if not pkl_files:
        logging.warning(f"⚠️ No files found in {pkl_dir}/target/ or {pkl_dir}/steps/")
        return

    logging.info(f"🔍 Scanning {len(target_files)} target and {len(step_files)} step files for missing photometry...")
    models_repaired = 0

    for idx, pkl_file in enumerate(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logging.warning(f"⚠️ Could not read {pkl_file.name}: {e}")
            continue
            
        # 1. Skip if atmosphere_raw is missing or empty
        raw_df = data.get('atmosphere_raw')
        if raw_df is None or raw_df.empty:
            print(f"⚠️ Skipping {pkl_file.name} - 'atmosphere_raw' is missing or empty.")
            continue
            
        # 2. Skip if it already has valid photometry
        if 'photometry' in data and data['photometry']:
            if 'bands' in data['photometry'] and len(data['photometry']['bands']) > 0:
                continue

        try:
            # 3. Re-instantiate the ExoremOut object using the raw DataFrame
            atm_out = ExoremOut(raw_df)
            
            # 4. Run your exact comprehensive photometry calculator from physics.py
            new_photometry = calculate_comprehensive_photometry(atm_out)
            
            # 5. Inject the newly calculated block back into the dictionary
            data['photometry'] = new_photometry
            
            # 6. Safely overwrite the .pkl file using the .tmp safeguard
            temp_file = pkl_file.with_suffix('.pkl.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
            temp_file.replace(pkl_file)
            
            models_repaired += 1
            
            # =========================================================
            # 📊 SANITY CHECK: Print sample values every 10 repairs
            # =========================================================
            if models_repaired % 10 == 1: 
                bands = new_photometry.get('bands', {})
                folder_type = pkl_file.parent.name.upper() # Will say TARGET or STEPS
                logging.info(f"\n🌟 SANITY CHECK | Repaired [{folder_type}] Model: {pkl_file.name}")
                logging.info(f"   Calculated {len(bands)} total photometric bands.")
                
                # Pick a few key filters to display
                sample_filters = ["JWST/NIRCam.F444W", "JWST/MIRI.F1000W", "Paranal/SPHERE.IRDIS_B_H"]
                display_keys = [f for f in sample_filters if f in bands]
                
                if not display_keys:
                    display_keys = list(bands.keys())[:4]
                    
                for key in display_keys:
                    flux_w = bands[key].get('flux_W_m2_um', 0.0)
                    flux_jy = bands[key].get('flux_Jy', 0.0)
                    logging.info(f"   -> {key: <25} | {flux_w:.3e} W/m²/µm  |  {flux_jy:.3e} Jy")
                logging.info("-" * 60)
            # =========================================================
            
        except Exception as e:
            logging.error(f"❌ Failed to repair {pkl_file.name}: {e}")
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink() # Clean up temp file on failure

    logging.info(f"\n✅ Successfully repaired photometry for {models_repaired} .pkl files across 'target' and 'steps'!")
    logging.info("🚀 You can now run `compile_grid.py` to seamlessly rebuild your HDF5 and CSV catalogs.")

if __name__ == "__main__":
    
    # Point this to your main output directory
    GRID_OUTPUT_DIR = "../outputs/grid_run"
    
    repair_photometry_in_pkls(target_dir=GRID_OUTPUT_DIR)