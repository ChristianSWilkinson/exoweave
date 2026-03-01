import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from exowrap.output import ExoremOut
from fuzzycore.constants import BAR_TO_PA, G_CONST

def build_master_profile(atm_out: ExoremOut, int_results: dict, p_link_bar: float) -> pd.DataFrame:
    """
    Combines atmosphere and interior profiles into a continuous master grid.
    
    This function anchors the atmospheric altitude to the interior's absolute 
    radius exactly at the linking pressure (P_link). It then interpolates all 
    thermodynamic variables (Temperature, Density, Gravity) onto a single, 
    continuous pressure grid.
    
    Args:
        atm_out (ExoremOut): The parsed exowrap atmosphere outputs.
        int_results (dict): The converged fuzzycore interior outputs.
        p_link_bar (float): The pressure boundary where the models link (in bar).
        
    Returns:
        pd.DataFrame: A continuous planetary profile sorted from top-of-atmosphere 
                      down to the core.
    """
    logging.info(f"Stitching profiles at P_link = {p_link_bar:.2f} bar")
    p_link_pa = p_link_bar * BAR_TO_PA

    # ==========================================
    # 1. EXTRACT ATMOSPHERE DATA
    # ==========================================
    # Levels (Boundaries)
    p_atm_lvl = atm_out.pressure_levels
    alt_atm_lvl = atm_out.altitude_profile
    
    # Layers (Cell Centers)
    p_atm_lay = atm_out.pressure_profile
    t_atm_lay = atm_out.temperature_profile
    g_atm_lay = atm_out.gravity
    molm_atm_lay = atm_out.mean_molar_mass
    rho_atm_lay = atm_out.density_profile

    # ==========================================
    # 2. EXTRACT INTERIOR DATA
    # ==========================================
    # fuzzycore outputs log10(P) in bars. Convert back to absolute Pascals.
    p_int_pa = (10 ** int_results['P']) * BAR_TO_PA
    r_int = int_results['R']
    t_int = int_results['T']
    rho_int = int_results['Rho']
    
    # Calculate interior gravity (g = GM/r^2)
    m_int = int_results['M']
    g_int = (G_CONST * m_int) / (r_int ** 2)

    # ==========================================
    # 3. SPATIAL STITCHING (RADIUS)
    # ==========================================
    # Atmosphere uses relative altitude (m); Interior uses absolute radius (m).
    # We find the exact physical radius at P_link and shift the atmosphere to match.
    
    int_rad_interp = interp1d(np.log10(p_int_pa), r_int, fill_value="extrapolate")
    atm_alt_interp = interp1d(np.log10(p_atm_lvl), alt_atm_lvl, fill_value="extrapolate")

    rad_at_plink = float(int_rad_interp(np.log10(p_link_pa)))
    alt_at_plink = float(atm_alt_interp(np.log10(p_link_pa)))
    
    logging.debug(f"Matching Alt={alt_at_plink:.2e} m to Rad={rad_at_plink:.2e} m")
    
    # Shift atmospheric altitude to make the spatial grid perfectly continuous
    r_atm_stitched = (alt_atm_lvl - alt_at_plink) + rad_at_plink

    # ==========================================
    # 4. MASTER GRID CREATION
    # ==========================================
    # Take atmosphere down to P_link, and interior below P_link
    atm_mask_lvl = p_atm_lvl <= p_link_pa
    int_mask = p_int_pa > p_link_pa
    
    master_p_pa = np.concatenate((p_atm_lvl[atm_mask_lvl], p_int_pa[int_mask]))
    master_r_m = np.concatenate((r_atm_stitched[atm_mask_lvl], r_int[int_mask]))
    
    logP_master = np.log10(master_p_pa)

    # ==========================================
    # 5. THERMODYNAMIC INTERPOLATION
    # ==========================================
    # Create interpolators mapping log(P) to the target variables
    atm_t_interp = interp1d(np.log10(p_atm_lay), t_atm_lay, fill_value="extrapolate")
    int_t_interp = interp1d(np.log10(p_int_pa), t_int, fill_value="extrapolate")
    
    atm_g_interp = interp1d(np.log10(p_atm_lay), g_atm_lay, fill_value="extrapolate")
    int_g_interp = interp1d(np.log10(p_int_pa), g_int, fill_value="extrapolate")
    
    atm_molm_interp = interp1d(np.log10(p_atm_lay), molm_atm_lay, fill_value="extrapolate")
    
    # Density varies by orders of magnitude, so we MUST interpolate in log-log space
    atm_rho_interp = interp1d(np.log10(p_atm_lay), np.log10(rho_atm_lay), fill_value="extrapolate")
    int_rho_interp = interp1d(np.log10(p_int_pa), np.log10(rho_int), fill_value="extrapolate")
    
    # Allocate final arrays
    master_t_k = np.zeros_like(master_p_pa)
    master_g_ms2 = np.zeros_like(master_p_pa)
    master_rho_kgm3 = np.zeros_like(master_p_pa)
    master_molm = np.full_like(master_p_pa, np.nan) # Deep interior won't have molecular mass
    
    # Apply interpolators to the master grid
    atm_mask = master_p_pa <= p_link_pa
    int_mask = master_p_pa > p_link_pa
    
    master_t_k[atm_mask] = atm_t_interp(logP_master[atm_mask])
    master_t_k[int_mask] = int_t_interp(logP_master[int_mask])
    
    master_g_ms2[atm_mask] = atm_g_interp(logP_master[atm_mask])
    master_g_ms2[int_mask] = int_g_interp(logP_master[int_mask])
    
    master_rho_kgm3[atm_mask] = 10 ** atm_rho_interp(logP_master[atm_mask])
    master_rho_kgm3[int_mask] = 10 ** int_rho_interp(logP_master[int_mask])
    
    master_molm[atm_mask] = atm_molm_interp(logP_master[atm_mask])

    # ==========================================
    # 6. FINALIZE AND SORT
    # ==========================================
    # Sort from Lowest Pressure (Top of Atmosphere) to Highest Pressure (Core)
    sort_idx = np.argsort(master_p_pa)

    return pd.DataFrame({
        'Pressure_Pa': master_p_pa[sort_idx],
        'Pressure_bar': master_p_pa[sort_idx] / BAR_TO_PA,
        'Radius_m': master_r_m[sort_idx],
        'Temperature_K': master_t_k[sort_idx],
        'Density_kgm3': master_rho_kgm3[sort_idx],
        'Gravity_ms2': master_g_ms2[sort_idx],
        'MolarMass_kgmol': master_molm[sort_idx]
    })