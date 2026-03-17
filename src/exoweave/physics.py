import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from exowrap.output import ExoremOut
from exowrap.photometry import get_svo_filter
from fuzzycore.constants import BAR_TO_PA, SIGMA_SB
from fuzzycore.utils import calculate_staircase_dt_ds
from fuzzycore.constants import G_CONST

def calculate_new_tint(
    atm_out, 
    pressure_threshold_bar: float = 10.0, 
    fallback_t_int: float = np.nan
) -> float:
    """
    Calculates the true Intrinsic Temperature (T_int) by isolating ExoREM's 
    explicit internal heat flux tracker (radiosity_internal) deep in the atmosphere.
    """
    logging.debug("Extracting true T_int from deep internal radiosity...")
    
    try:
        df = atm_out.df
        p_pa = np.asarray(df['/outputs/levels/pressure'].iloc[0])
        
        # ExoREM perfectly isolates the net internal heat flux in this array!
        rad_int = np.asarray(df['/outputs/levels/radiosity_internal'].iloc[0])
        
        # Mask for the deep atmosphere to avoid upper atmospheric noise
        p_threshold_pa = pressure_threshold_bar * 1e5
        deep_mask = p_pa >= p_threshold_pa
        
        if np.any(deep_mask):
            avg_internal_flux = np.nanmean(rad_int[deep_mask])
            
            if avg_internal_flux > 0:
                t_int_comp = (avg_internal_flux / 5.67e-8) ** 0.25
                return float(t_int_comp)

        # Only used if the entire array is negative/corrupted
        return fallback_t_int

    except Exception as e:
        logging.error(f"⚠️ Error verifying T_int: {e}. Falling back to dial value.")
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
    """
    
    logging.info(f"Calculating Z_base and Y_ratio from atmospheric VMRs at {p_link_bar} bar.")
    
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
        
        all_vmrs = atm_out.vmr_profiles
        vmr_h2, vmr_he, vmr_h = 0.0, 0.0, 0.0
        
        # Safely extract VMRs using the exact path
        # This completely avoids the 'elements_gas_phase' atomic counters!
        for key, vmr_array in all_vmrs.items():
            if key == 'gas:H2':
                vmr_h2 = vmr_array[lay_idx]
            elif key == 'gas:He':
                vmr_he = vmr_array[lay_idx]
            elif key == 'gas:H':
                vmr_h = vmr_array[lay_idx]
                
        # Calculate Absolute Mass Fractions (X = Hydrogen, Y = Helium)
        X = ((vmr_h2 * M_H2) + (vmr_h * M_H)) / mu
        Y = (vmr_he * M_He) / mu
        raw_Z = 1.0 - (X + Y)
        
        # Clean numerical noise (ensure it's between 0 and 0.99)
        Z = max(min(raw_Z, 0.99), 0.0)
        
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

# ---------------------------------------------------------
# GLOBAL FILTER LIST & DISK CACHE SETUP
# ---------------------------------------------------------
TARGET_FILTERS = [
    # --- JWST ---
    "JWST/NIRCam.F070W", "JWST/NIRCam.F090W", "JWST/NIRCam.F115W", "JWST/NIRCam.F140M", 
    "JWST/NIRCam.F150W", "JWST/NIRCam.F182M", "JWST/NIRCam.F200W", "JWST/NIRCam.F210M", 
    "JWST/NIRCam.F250M", "JWST/NIRCam.F277W", "JWST/NIRCam.F300M", "JWST/NIRCam.F335M", 
    "JWST/NIRCam.F356W", "JWST/NIRCam.F410M", "JWST/NIRCam.F430M", "JWST/NIRCam.F444W", 
    "JWST/NIRCam.F460M", "JWST/NIRCam.F480M",
    "JWST/MIRI.F560W", "JWST/MIRI.F770W", "JWST/MIRI.F1000W", "JWST/MIRI.F1130W", 
    "JWST/MIRI.F1280W", "JWST/MIRI.F1500W", "JWST/MIRI.F1800W", "JWST/MIRI.F2100W", "JWST/MIRI.F2550W",
    "JWST/NIRISS.F090W", "JWST/NIRISS.F115W", "JWST/NIRISS.F140M", "JWST/NIRISS.F150W", 
    "JWST/NIRISS.F158M", "JWST/NIRISS.F200W", "JWST/NIRISS.F277W", "JWST/NIRISS.F380M", 
    "JWST/NIRISS.F430M", "JWST/NIRISS.F480M",
    
    # --- VLT (Paranal) ---
    "Paranal/SPHERE.IRDIS_B_Y", "Paranal/SPHERE.IRDIS_B_J", "Paranal/SPHERE.IRDIS_B_H", "Paranal/SPHERE.IRDIS_B_Ks",
    "Paranal/SPHERE.IRDIS_D_J23_2", "Paranal/SPHERE.IRDIS_D_J23_3", 
    "Paranal/SPHERE.IRDIS_D_H23_2", "Paranal/SPHERE.IRDIS_D_H23_3", 
    "Paranal/SPHERE.IRDIS_D_K12_1", "Paranal/SPHERE.IRDIS_D_K12_2",
    "Paranal/NACO.J", "Paranal/NACO.H", "Paranal/NACO.Ks", "Paranal/NACO.Lp", "Paranal/NACO.Mp",
    "Paranal/HAWKI.J", "Paranal/HAWKI.H", "Paranal/HAWKI.Ks", "Paranal/HAWKI.CH4",
    "Paranal/VISIR.B8_7", "Paranal/VISIR.B10_7", "Paranal/VISIR.B11_7", "Paranal/VISIR.Q2",
    
    # --- KECK & GEMINI ---
    "Keck/NIRC2.J", "Keck/NIRC2.H", "Keck/NIRC2.Ks", "Keck/NIRC2.Kp", "Keck/NIRC2.Lp", "Keck/NIRC2.Ms",
    "Gemini/NIRI.J", "Gemini/NIRI.H", "Gemini/NIRI.K", "Gemini/NIRI.L-prime", "Gemini/NIRI.M-prime",
    
    # --- SPACE: HST, SPITZER, WISE ---
    "HST/WFC3_IR.F110W", "HST/WFC3_IR.F140W", "HST/WFC3_IR.F160W", "HST/WFC3_UVIS1.F606W", "HST/WFC3_UVIS1.F814W",
    "Spitzer/IRAC.I1", "Spitzer/IRAC.I2", "Spitzer/IRAC.I3", "Spitzer/IRAC.I4",
    "WISE/WISE.W1", "WISE/WISE.W2", "WISE/WISE.W3", "WISE/WISE.W4",
    
    # --- NEXT-GEN SPACE: ROMAN ---
    "Roman/WFI.F062", "Roman/WFI.F087", "Roman/WFI.F106", "Roman/WFI.F129", 
    "Roman/WFI.F146", "Roman/WFI.F158", "Roman/WFI.F184",
    
    # --- ALL-SKY SURVEYS: 2MASS, SDSS, PAN-STARRS, GAIA, TESS ---
    "2MASS/2MASS.J", "2MASS/2MASS.H", "2MASS/2MASS.Ks",
    "SLOAN/SDSS.u", "SLOAN/SDSS.g", "SLOAN/SDSS.r", "SLOAN/SDSS.i", "SLOAN/SDSS.z",
    "PAN-STARRS/PS1.g", "PAN-STARRS/PS1.r", "PAN-STARRS/PS1.i", "PAN-STARRS/PS1.z", "PAN-STARRS/PS1.y",
    "GAIA/GAIA3.G", "GAIA/GAIA3.Gbp", "GAIA/GAIA3.Grp",
    "TESS/TESS.Red", "Kepler/Kepler.K"
]

# Physical file shared by all CPU cores
CACHE_FILE = Path(__file__).resolve().parent / "svo_filter_cache.pkl"
_FILTER_CACHE = {}

# Load the disk cache into memory as soon as a CPU core imports this file
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, 'rb') as f:
            _FILTER_CACHE = pickle.load(f)
    except Exception:
        pass

def calculate_comprehensive_photometry(atm_out) -> dict:
    """
    Extracts raw spectral arrays and computes synthetic photometry. 
    Relies on a pre-warmed shared disk cache to prevent SVO Server bans.
    """
    logging.info("🌟 Computing Mega-Catalog Photometry (Using Disk Cache)...")
    
    exo_wl = getattr(atm_out, 'wavelength', None)
    exo_flux = getattr(atm_out, 'flux_flambda', None)
    
    photometry_section = {
        'wavelength_um': exo_wl,
        'emission_flux_W_m2_um': exo_flux,
        'transit_depth': getattr(atm_out, 'transit_depth', None), 
        'bands': {}
    }
    
    if exo_wl is None or exo_flux is None:
        logging.error("Missing raw spectral arrays; skipping photometry.")
        return photometry_section
        
    valid = exo_wl > 0
    exo_wl = exo_wl[valid]
    exo_flux = exo_flux[valid]

    sort_idx = np.argsort(exo_wl)
    exo_wl = exo_wl[sort_idx]
    exo_flux = exo_flux[sort_idx]

    # Process all filters
    for filter_id in TARGET_FILTERS:
        try:
            if filter_id not in _FILTER_CACHE:
                # Fallback: Fetch it and gently update the memory dict
                # (This will rarely trigger if run_grid pre-warms the cache properly)
                _FILTER_CACHE[filter_id] = get_svo_filter(filter_id)
                
            filt_wav, filt_trans = _FILTER_CACHE[filter_id]
            
            # --- INTEGRATION MATH ---
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

            photometry_section['bands'][filter_id] = {
                "filter_id": filter_id,
                "effective_wavelength_um": eff_wav,
                "flux_W_m2_um": phot_flux_flambda,
                "flux_Jy": phot_flux_jy
            }
            
        except Exception as e:
            # Now logs the error quietly instead of silently hiding server bans
            logging.debug(f"Skipping {filter_id}: {e}")

    logging.info(f"✅ Successfully computed {len(photometry_section['bands'])} photometric bands!")
    return photometry_section