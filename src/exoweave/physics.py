import logging
import numpy as np
import pandas as pd

from exowrap.output import ExoremOut
from fuzzycore.constants import BAR_TO_PA, SIGMA_SB
from fuzzycore.utils import calculate_staircase_dt_ds
from fuzzycore.constants import G_CONST

def calculate_new_tint(
    atm_out: ExoremOut, 
    pressure_threshold_bar: float = 10.0, 
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


def calculate_z_base(atm_out, p_link_bar: float, fallback_met: float = 0.0) -> tuple[float, float]:
    """
    Calculates the heavy element mass fraction (Z) and the Helium-to-(Hydrogen+Helium) 
    mass ratio (Y_ratio) at the linking boundary.
    
    Args:
        atm_out (ExoremOut): The parsed ExoREM atmosphere outputs.
        p_link_bar (float): The pressure boundary (in bar) to extract from.
        fallback_met (float): The input metallicity [Fe/H] to use as a fallback 
                              if VMR extraction fails.
                              
    Returns:
        tuple[float, float]: A tuple containing (Z, Y_ratio).
    """
    import logging
    import numpy as np
    
    logging.debug(f"Calculating Z_base and Y_ratio from atmospheric VMRs at {p_link_bar} bar.")
    
    try:
        # Find the atmospheric layer closest to the linking pressure
        p_layers_bar = atm_out.pressure_profile / 1e5
        lay_idx = np.argmin(np.abs(p_layers_bar - p_link_bar))
        
        # Total mean molar mass of the mixture at this layer (kg/mol)
        mu = atm_out.mean_molar_mass[lay_idx]
        
        # Standard Molar Masses (kg/mol)
        M_H2 = 2.01588e-3
        M_He = 4.00260e-3
        M_H  = 1.00784e-3
        
        # Retrieve all VMR profiles (absorbers, gases, etc.)
        all_vmrs = atm_out.vmr_profiles
        
        vmr_h2, vmr_he, vmr_h = 0.0, 0.0, 0.0
        
        # Safely extract H2, He, and H regardless of their dictionary prefix
        for key, vmr_array in all_vmrs.items():
            if key.endswith(':H2') or key == 'H2':
                vmr_h2 = vmr_array[lay_idx]
            elif key.endswith(':He') or key == 'He':
                vmr_he = vmr_array[lay_idx]
            elif key.endswith(':H') or key == 'H':
                vmr_h = vmr_array[lay_idx]
                
        # Calculate Absolute Mass Fractions (X = Hydrogen, Y = Helium)
        # Equation: Mass_Fraction = (VMR * Molar_Mass_Species) / Mean_Molar_Mass
        X = ((vmr_h2 * M_H2) + (vmr_h * M_H)) / mu
        Y = (vmr_he * M_He) / mu
        
        # Z is the mass fraction of all remaining heavy elements
        Z = 1.0 - (X + Y)
        
        # Clean numerical noise (ensure it's between 0 and 0.99)
        Z = max(min(Z, 0.99), 0.0)
        
        # If VMRs were totally missing (Z ≈ 1.0), trigger the fallback
        if Z > 0.98 and vmr_h2 == 0.0:
            raise ValueError("H2/He VMRs not found in ExoREM output.")
            
        # Calculate the internal Y_ratio (Y / (X + Y)) for the EOS mixer
        if (X + Y) > 0:
            Y_ratio = Y / (X + Y)
        else:
            Y_ratio = 0.26 # Fallback to proto-solar if atmosphere is purely heavy elements
            
        logging.info(f"🧪 Chemical Sync: Derived Z_base = {Z:.4f}, Y_ratio = {Y_ratio:.4f} (from X={X:.4f}, Y={Y:.4f})")
        return float(Z), float(Y_ratio)

    except Exception as e:
        # Fallback: Approximate Z from the bulk atmospheric metallicity input
        fallback_z = min(0.015 * (10 ** fallback_met), 0.99)
        fallback_y_ratio = 0.26 # Proto-solar fallback
        logging.warning(f"⚠️ Failed to extract Z and Y_ratio from VMRs ({e}). Using fallbacks: Z ≈ {fallback_z:.4f}, Y_ratio = {fallback_y_ratio}")
        return float(fallback_z), float(fallback_y_ratio)


def calculate_stitched_mass(atm_out, int_results: dict, p_link_bar: float) -> tuple[float, float, float]:
    """
    Calculates the precise total planetary mass by integrating the atmospheric 
    envelope's mass shell-by-shell and adding it to the interior core mass.
    
    Args:
        atm_out (ExoremOut): The parsed ExoREM atmosphere outputs.
        int_results (dict): The converged fuzzycore interior results.
        p_link_bar (float): The pressure boundary (in bar) where the models link.
        
    Returns:
        tuple: (total_mass_kg, interior_mass_kg, atm_mass_kg)
    """
    
    # 1. Extract the interior properties exactly at the junction
    interior_mass_kg = int_results['M'][-1]
    r_link_m = int_results['R'][-1]
    
    # 2. Get the atmospheric layer arrays natively from ExoREM
    p_layers_pa = atm_out.pressure_profile
    rho_layers = atm_out.density_profile
    
    # 3. Filter arrays to only include the atmosphere ABOVE the junction
    mask = p_layers_pa <= (p_link_bar * 1e5)
    p_atm = p_layers_pa[mask]
    rho_atm = rho_layers[mask]
    
    # Sort from bottom (highest pressure) to top (lowest pressure)
    sort_idx = np.argsort(p_atm)[::-1]
    p_atm = p_atm[sort_idx]
    rho_atm = rho_atm[sort_idx]
    
    # 4. Generate pressure boundaries (edges) for each layer
    edges = [p_link_bar * 1e5]
    for i in range(len(p_atm) - 1):
        edges.append((p_atm[i] + p_atm[i+1]) / 2.0)
    edges.append(p_atm[-1] * 0.5)
    
    # 5. Numerically integrate dm and dr shell-by-shell up to space
    m_current = interior_mass_kg
    r_current = r_link_m
    m_atm_kg = 0.0
    
    for i in range(len(p_atm)):
        p_bottom = edges[i]
        p_top = edges[i+1]
        dp = p_top - p_bottom  # Negative value
        
        rho_layer = rho_atm[i]
        
        # Calculate thickness and mass of this shell
        dr = -dp * (r_current**2) / (rho_layer * G_CONST * m_current)
        dm = 4 * np.pi * (r_current**2) * rho_layer * dr
        
        r_current += dr
        m_current += dm
        m_atm_kg += dm
        
    return m_current, interior_mass_kg, m_atm_kg