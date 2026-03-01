import logging
import pickle as pkl
from pathlib import Path
from datetime import datetime
import pandas as pd

def _generate_filename(params: dict, suffix: str = "") -> str:
    """
    Generates a consistent filename based on the physical parameters.
    Replaces the old _generate_param_string method.
    """
    mass = params.get('mass', 'NaN')
    t_int = params.get('T_int', 'NaN')
    met = params.get('Met', 'NaN')
    
    # Format: M_1.00_Tint_500.0_Met_0.00
    base = f"M_{mass:.2f}_Tint_{t_int:.1f}_Met_{met:.2f}"
    
    if suffix:
        return f"{base}_{suffix}.pkl"
    return f"{base}.pkl"

def save_converged_model(results: dict, output_dir: str | Path, custom_name: str = None) -> Path:
    """
    Saves a converged planetary model to disk. 
    
    Instead of stuffing arrays into a DataFrame cell, it saves a structured 
    dictionary containing the clean stitched DataFrame, the raw outputs, 
    and the parameters.
    
    Args:
        results (dict): The output dictionary from ExoCoupler.run().
        output_dir (str | Path): The directory to save the file in.
        custom_name (str, optional): Override the default generated filename.
        
    Returns:
        Path: The exact path where the file was saved.
    """
    out_dir = Path(output_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    params = results.get('final_params', {})
    filename = custom_name if custom_name else _generate_filename(params)
    filepath = out_dir / filename
    
    # Structure the data cleanly
    data_to_save = {
        'status': 'converged',
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'iterations': results.get('iterations'),
        'profile': results.get('stitched_profile'), # The clean pd.DataFrame
        'atmosphere_raw': results.get('atmosphere_raw'),
        'interior_raw': results.get('interior_raw')
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
    Logs and saves the parameter history of a failed run for debugging.
    Replaces the old _handle_simulation_failure method.
    
    Args:
        history (dict): The iteration history dictionary from the coupler.
        params (dict): The target physical parameters.
        reason (str): Why the simulation failed (e.g., "Max iterations reached").
        output_dir (str | Path): The root output directory.
        
    Returns:
        Path: The exact path where the failure log was saved.
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