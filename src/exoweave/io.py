import logging
import pickle as pkl
from pathlib import Path
from datetime import datetime

def _generate_filename(params: dict, actual_mass: float = None, suffix: str = "") -> str:
    """
    Generates an exhaustive, non-confusable filename based on the physical parameters.
    If actual_mass is provided (for intermediate steps), it overrides the target mass.
    """
    mass = actual_mass if actual_mass is not None else params.get('mass', 0.0)
    t_irr = params.get('T_irr', 0.0)
    t_int = params.get('T_int', 0.0)
    met = params.get('Met', 0.0)
    core = params.get('core_mass_earth', 0.0)
    fsed = params.get('f_sed', 0.0)
    kzz = params.get('kzz', 0.0)
    sigma = params.get('sigma_val', 0.0)
    
    # Format: M_1.000_Tirr_100.0_Tint_500.0_Met_0.00_Core_15.0_fsed_1.0_kzz_8.0_sigma_0.05
    base = (f"M_{mass:.3f}_Tirr_{t_irr:.1f}_Tint_{t_int:.1f}_"
            f"Met_{met:.2f}_Core_{core:.1f}_fsed_{fsed:.1f}_kzz_{kzz:.1f}_sigma_{sigma:.2f}")
    
    if suffix:
        return f"{base}_{suffix}.pkl"
    return f"{base}.pkl"

def save_step_model(step_data: dict, output_dir: str | Path) -> Path:
    """
    Saves an intermediate, valid iteration step to the 'steps' subfolder.
    """
    out_dir = Path(output_dir) / "steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    params = step_data.get('parameters', {})
    iteration = step_data.get('iteration', 0)
    actual_mass = step_data.get('mass_calculated_mjup')
    
    filename = _generate_filename(params, actual_mass=actual_mass, suffix=f"iter_{iteration}")
    filepath = out_dir / filename
    
    try:
        # Step models already dump the full dictionary provided by coupler.py
        with open(filepath, 'wb') as f:
            pkl.dump(step_data, f)
        logging.debug(f"💾 Step {iteration} saved to: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"❌ Failed to save step model: {e}", exc_info=True)
        return None

def save_converged_model(results: dict, output_dir: str | Path, custom_name: str = None) -> Path:
    """
    Saves a finalized, converged planetary model to the 'target' subfolder.
    """
    out_dir = Path(output_dir) / "target"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    params = results.get('final_params', {})
    filename = custom_name if custom_name else _generate_filename(params)
    filepath = out_dir / filename
    
    # 🚨 FIXED: Now explicitly identical in structure to the step_data dictionary
    data_to_save = {
        'status': 'converged',
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'iterations': results.get('iterations'),
        'profile': results.get('stitched_profile'), 
        'atmosphere_raw': results.get('atmosphere_raw'),
        'interior_raw': results.get('interior_raw'),
        'photometry': results.get('photometry'),
        'cooling_metrics': results.get('cooling_metrics')
    }
    
    try:
        with open(filepath, 'wb') as f:
            pkl.dump(data_to_save, f)
        logging.info(f"💾 Converged model saved successfully to: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"❌ Failed to save converged model to {filepath}: {e}", exc_info=True)
        return None

def save_failed_run(history: dict, params: dict, reason: str, output_dir: str | Path) -> Path:
    """
    Logs and saves the parameter history of a failed run into a 'failed' subfolder.
    """
    fail_dir = Path(output_dir) / "failed"
    fail_dir.mkdir(parents=True, exist_ok=True)
    
    filename = _generate_filename(params, suffix="FAILED")
    filepath = fail_dir / filename
    
    failure_data = {
        'status': 'failed',
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'failure_reason': reason,
        'history': history
    }
    
    try:
        with open(filepath, 'wb') as f:
            pkl.dump(failure_data, f)
        logging.info(f"⚠️ Failure log saved to: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"❌ Failed to save failure log to {filepath}: {e}", exc_info=True)
        return None