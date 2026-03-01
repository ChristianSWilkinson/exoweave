import logging
import numpy as np
import pandas as pd

from exowrap.output import ExoremOut
from fuzzycore.constants import BAR_TO_PA, SIGMA_SB
from fuzzycore.utils import calculate_staircase_dt_ds

def calculate_new_tint(
    atm_out: ExoremOut, 
    pressure_threshold_bar: float = 100.0, 
    fallback_t_int: float = 500.0
) -> float:
    """
    Calculates the Intrinsic Temperature (T_int) based on the average 
    upward thermal flux within the deepest convective zone.
    """
    logging.debug("Calculating T_int from average flux in deepest convective zone.")
    
    try:
        p_pa = np.asarray(atm_out.df['/outputs/levels/pressure'].iloc[0])
        is_conv = np.asarray(atm_out.df['/outputs/levels/is_convective'].iloc[0]).astype(bool)
        rad_int = np.asarray(atm_out.df['/outputs/levels/radiosity_internal'].iloc[0])
        rad_conv = np.asarray(atm_out.df['/outputs/levels/radiosity_convective'].iloc[0])
        
        p_threshold_pa = pressure_threshold_bar * BAR_TO_PA
        
        candidate_indices = np.where((is_conv) & (p_pa >= p_threshold_pa))[0]

        if candidate_indices.size == 0:
            return fallback_t_int

        convective_zones = []
        current_zone_start = -1
        
        for i in range(len(p_pa)):
            if i in candidate_indices:
                if current_zone_start == -1:
                    current_zone_start = i
            else:
                if current_zone_start != -1:
                    convective_zones.append((current_zone_start, i - 1))
                    current_zone_start = -1
                    
        if current_zone_start != -1:
            convective_zones.append((current_zone_start, len(p_pa) - 1))
            
        if not convective_zones:
            return fallback_t_int
            
        deepest_zone = max(convective_zones, key=lambda zone: zone[0])
        start_idx, end_idx = deepest_zone

        total_flux_profile = rad_int + rad_conv
        flux_in_zone = total_flux_profile[start_idx : end_idx + 1]
        
        avg_flux = np.nanmean(flux_in_zone)
        
        if pd.notna(avg_flux) and avg_flux >= 0:
            t_int_comp = (avg_flux / SIGMA_SB) ** 0.25
            logging.info(f"Calculated T_int: {t_int_comp:.2f} K (Avg Deep Flux = {avg_flux:.3e} W/m²)")
            return float(t_int_comp)
        else:
            return fallback_t_int

    except Exception as e:
        logging.error(f"Error calculating T_int from convective zone: {e}", exc_info=True)
        return fallback_t_int


def calculate_entropy_evolution(int_results: dict, t_int: float) -> dict:
    """
    Calculates the intrinsic luminosity (L_int) and the rate of change of 
    specific entropy (ds/dt) using fuzzycore's native staircase integration.
    
    Args:
        int_results (dict): The converged interior structure from fuzzycore.
        t_int (float): The converged intrinsic temperature (K).
        
    Returns:
        dict: Contains 'ds_dt', 'L_int', 'dt_ds', and layer contributions.
    """
    logging.info("Calculating entropy evolution using fuzzycore native utility.")
    
    # 1. Call fuzzycore's native dt/dS calculator
    cooling_data = calculate_staircase_dt_ds(results=int_results, t_int=t_int)
    
    total_dt_ds = cooling_data['total_dt_ds']
    
    # 2. Invert to get ds/dt (Entropy decreases as the planet cools, hence negative)
    if total_dt_ds > 0 and not np.isinf(total_dt_ds):
        ds_dt = -1.0 / total_dt_ds
    else:
        logging.warning("fuzzycore returned invalid dt/dS. Setting ds/dt to NaN.")
        ds_dt = np.nan
        
    # 3. Calculate Intrinsic Luminosity for completeness
    r_planet = int_results['R'][-1]
    l_int = 4 * np.pi * (r_planet ** 2) * SIGMA_SB * (t_int ** 4)

    logging.info(f"Calculated ds/dt: {ds_dt:.3e} W/K | L_int: {l_int:.3e} W")
    
    return {
        'ds_dt': float(ds_dt),
        'dt_ds': float(total_dt_ds),
        'L_int': float(l_int),
        'layer_contributions': cooling_data.get('layer_contributions', {})
    }