"""
ExoWeave Coupler Module.

Orchestrates the iterative coupling between exowrap (Atmosphere) and 
fuzzycore (Interior) models. Uses a 1D secant root-finding method to adjust 
surface gravity until the fully integrated atmospheric + interior mass 
matches the target planetary mass.
"""

import logging
import os
import pickle
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

# Resolve the absolute path to the exoweave project root
EXOWEAVE_ROOT = Path(__file__).resolve().parents[2]

# External dependencies
from exowrap.model import Simulation
from exowrap.output import ExoremOut
from exowrap.tools import upgrade_resolution
from fuzzycore.constants import G_CONST, M_JUPITER, R_JUPITER
from fuzzycore.solver import solve_structure
from fuzzycore.utils import DummyLock, generate_gaussian_z_profile
import fuzzycore.constants as c

# Internal exoweave modules
from .io import save_converged_model, save_failed_run, save_step_model
from .physics import (
    calculate_comprehensive_photometry,
    calculate_new_tint,
    calculate_stitched_mass,
    calculate_z_base,
)
from .profile import build_master_profile


class ExoCoupler:
    """
    Orchestrates the iterative coupling between atmospheric and interior models.
    """

    def __init__(self, target_params: dict, config: dict):
        """
        Initializes the coupler with target parameters and grid configuration.

        Args:
            target_params (dict): Physical parameters for the target planet.
            config (dict): Grid execution and convergence settings.
        """
        self.params = target_params.copy()
        self.config = config
        
        # --- Output Directory Setup ---
        raw_out_dir = config.get("output_dir", "exoweave_outputs")
        out_path = Path(raw_out_dir)
        
        if not out_path.is_absolute():
            self.output_dir = EXOWEAVE_ROOT / out_path
        else:
            self.output_dir = out_path
            
        self.tmp_dir = EXOWEAVE_ROOT / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.tmp_dir / f"solver_steps_{os.getpid()}.csv"
        
        # --- Convergence Settings ---
        self.max_iterations = config.get("max_iterations", 15)
        self.mass_tol = config.get("mass_convergence_threshold", 0.01)
        self.p_bottom_bar = config.get("p_bottom_bar", 1000.0) 
        self.p_link_bar = config.get("p_link_target_bar", self.p_bottom_bar)
        self.min_p_link_bar = config.get("min_p_link_bar", 0.1)
        
        self.history = {
            'iteration': [], 
            'T_int_measured': [], 
            'g_1bar': [], 
            'mass_calculated': [], 
            'mass_error': []
        }

    def _guess_initial_gravity(self) -> float:
        """
        Calculates a smart initial gravity guess based on an empirical 
        Mass-Radius prior to speed up convergence.
        
        Returns:
            float: Initial surface gravity guess in m/s^2.
        """
        mass_kg = self.params['mass'] * M_JUPITER
        radius_guess = R_JUPITER 
        
        if self.params['mass'] < 0.5:
            radius_guess = R_JUPITER * (self.params['mass'] ** 0.3)
            
        initial_g = (G_CONST * mass_kg) / (radius_guess ** 2)
        logging.info(
            f"💡 Smart Initialization: Guessed g = {initial_g:.2f} m/s² "
            f"for M = {self.params['mass']} M_Jup"
        )
        return float(initial_g)
    
    def _bootstrap_gravity(self) -> float:
        """
        Runs a rapid, 1-bar interior solve using a fully convective (Sharp Core) 
        assumption to guarantee a physically stable initial gravity guess.
        """
        logging.info("👢 Bootstrapping initial gravity via 1-bar interior pre-solve...")
        
        target_mass_kg = self.params['mass'] * M_JUPITER
        
        # --- 1. INTEGRATE DIRECTLY TO 1 BAR ---
        p_surf_pa = 1.0  # Exactly 1 bar
        t_surf = max(self.params.get('T_irr', 500.0) * 0.7, self.params.get('T_int', 500.0))
        met_dial = self.params.get('Met', 0.0)
        z_guess = min(0.015 * (10 ** met_dial), 0.95)
        
        z_profile = generate_gaussian_z_profile(
            n_layers=1, 
            sigma=0.0, 
            z_base=z_guess, 
            z_core=z_guess
        )
        
        # Dynamic starting guess to prevent Gas Giant roots for Sub-Neptunes
        mass_mj = self.params['mass']
        if mass_mj < 0.05: log_pc_guess = 6.8  # < 15 Earth masses
        elif mass_mj < 0.1: log_pc_guess = 7.5 # 15 - 30 Earth masses
        elif mass_mj < 0.5: log_pc_guess = 8.5 # Saturns
        elif mass_mj <= 2.0: log_pc_guess = 9.5 # Jupiters
        else: log_pc_guess = 10.5 # Super-Jupiters

        fc_params = {
            'P_surf': p_surf_pa,
            'T_surf': t_surf,
            'M_core': self.params.get('core_mass_earth', 10.0) * c.M_EARTH, 
            'M_water':  self.params.get('M_water', 0.0) * c.M_EARTH, 
            'iron_fraction': self.params.get('iron_fraction', 0.33),
            'z_base': z_guess,                        # <--- NOW DYNAMIC
            'Y_ratio': 0.26,                          
            'sigma_val': 0.0,  
            'z_profile': z_profile,                   # <--- NOW DYNAMIC
            'initial_log_pc': log_pc_guess,                      
            'debug': self.params.get('debug', False)    
        }


        
        # --- 3. SOLVE ---        
        int_results = solve_structure(
            target_val=target_mass_kg,
            params=fc_params,
            mode='mass',                              
            trial_id="bootstrap",
            csv_file=os.devnull,
            write_lock=DummyLock()
        )
        
        if int_results is None:
            logging.warning("⚠️ Bootstrap interior failed. Falling back to empirical guess.")
            return self._guess_initial_gravity()
            
        # --- 4. EXACT 1-BAR GRAVITY ---
        # Because we integrated to 1 bar, the final radius is exactly R_1bar!
        r_1bar_m = int_results['R'][-1]
        interior_mass_kg = int_results['M'][-1]
        
        calc_g = (G_CONST * interior_mass_kg) / (r_1bar_m ** 2)
        
        # --- NEW: BOOTSTRAP CLAMP ---
        if calc_g < 1.5:
            logging.warning(f"⚠️ Bootstrap yielded dangerously low g = {calc_g:.2f} m/s²! Clamping to 1.0 m/s².")
            calc_g = 1.5
        
        logging.info(f"✅ Bootstrap successfully locked initial g_1bar = {calc_g:.2f} m/s²")
        return float(calc_g)

    def _extract_boundary_conditions(self, atm_out: ExoremOut) -> tuple:
        """
        Dynamically locates the junction point by identifying the thickest 
        continuous convective block in the atmosphere.
        
        Args:
            atm_out (ExoremOut): The parsed atmospheric model output.
            
        Returns:
            tuple: (Pressure in bar, Temperature in K) at the linking boundary.
        """
        p_levels_bar = atm_out.pressure_levels / 1e5
        t_levels = atm_out.temperature_levels
        
        try:
            is_conv_raw = atm_out._get('/outputs/levels/is_convective')
            is_conv = np.asarray(is_conv_raw).astype(bool)
        except Exception as e:
            logging.warning(
                f"Could not find convective flags ({e}). "
                "Falling back to static target."
            )
            idx_link = np.argmin(np.abs(p_levels_bar - self.p_link_bar))
            return float(p_levels_bar[idx_link]), float(t_levels[idx_link])

        sort_idx = np.argsort(p_levels_bar)
        p_sorted = p_levels_bar[sort_idx]
        t_sorted = t_levels[sort_idx]
        conv_sorted = is_conv[sort_idx]
        
        blocks = []
        in_block = False
        start_idx = 0
        
        for i, val in enumerate(conv_sorted):
            if val and not in_block:
                in_block = True
                start_idx = i
            elif not val and in_block:
                in_block = False
                blocks.append((start_idx, i - 1))
                
        if in_block:
            blocks.append((start_idx, len(conv_sorted) - 1))

        if not blocks:
            logging.warning("No convective regions found! Fallback to static.")
            idx_link = np.argmin(np.abs(p_levels_bar - self.p_link_bar))
            return float(p_levels_bar[idx_link]), float(t_levels[idx_link])

        best_block = None
        max_span = -1.0
        
        for start, end in blocks:
            span = np.log10(p_sorted[end] / p_sorted[start])
            if span > max_span:
                max_span = span
                best_block = (start, end)

        top_idx = best_block[0]
        candidate_p = float(p_sorted[top_idx])
        
        if candidate_p < self.min_p_link_bar:
            logging.info(
                f"🛡️ Convective top at {candidate_p:.3e} bar is too shallow. "
                f"Clamping to >= {self.min_p_link_bar} bar."
            )
            # p_sorted is ascending, so we find the first index >= our minimum
            valid_indices = np.where(p_sorted >= self.min_p_link_bar)[0]
            if len(valid_indices) > 0:
                top_idx = valid_indices[0]
            else:
                # Fallback if the whole atmosphere is somehow extremely low pressure
                top_idx = len(p_sorted) - 1

        self.p_link_bar = float(p_sorted[top_idx])
        
        logging.info(
            f"🔗 Dynamic Junction: Anchoring to thickest convective block "
            f"at P = {self.p_link_bar:.2f} bar"
        )
        return float(p_sorted[top_idx]), float(t_sorted[top_idx])
    
    def _find_closest_prior_profile(self, init_pt_file: Path) -> bool:
        """
        Scans the output directory for previously saved models to use as a 
        warm-started prior, drastically reducing the initial iteration time.
        
        Args:
            init_pt_file (Path): Destination to write the prior P-T profile.
            
        Returns:
            bool: True if a valid prior was found and written, False otherwise.
        """
        best_file = None
        best_distance = float('inf')
        threshold = 0.30  
        
        for pkl_path in self.output_dir.glob("**/*.pkl"):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    
                if 'parameters' not in data or 'profile' not in data:
                    continue
                    
                saved = data['parameters']
                
                m_diff = abs(saved.get('mass', 1.0) - self.params['mass']) / self.params['mass']
                t_int_diff = abs(saved.get('T_int', 500) - self.params['T_int']) / max(self.params['T_int'], 1)
                t_irr_diff = abs(saved.get('T_irr', 500) - self.params.get('T_irr', 500)) / max(self.params.get('T_irr', 500), 1)
                met_diff = abs(saved.get('Met', 0.0) - self.params.get('Met', 0.0))
                sigma_diff = abs(saved.get('sigma_val', 0.0) - self.params.get('sigma_val', 0.0))
                
                dist = np.sqrt(m_diff**2 + t_int_diff**2 + t_irr_diff**2 + met_diff**2 + sigma_diff**2)
                
                if dist < best_distance and dist < threshold:
                    best_distance = dist
                    best_file = pkl_path
                    best_data = data
                    
            except Exception:
                continue
                
        if best_file is not None:
            logging.info(
                f"🧠 Smart Prior: Found neighbor model ({best_file.name}) "
                f"with parameter distance {best_distance:.2f}"
            )
            df = best_data['profile']
            p_pa = df['Pressure_bar'].values * 1e5
            t_k = df['Temperature_K'].values
            
            target_p_bottom = self.p_bottom_bar * 1e5
            
            if p_pa[0] > 0.1:
                p_pa = np.insert(p_pa, 0, 0.1)
                t_k = np.insert(t_k, 0, t_k[0]) 
                
            if p_pa[-1] < target_p_bottom:
                p_pa = np.append(p_pa, target_p_bottom)
                t_bottom = t_k[-1] * (target_p_bottom / p_pa[-2]) ** 0.286
                t_k = np.append(t_k, t_bottom)
                
            if len(p_pa) > 200:
                idx = np.linspace(0, len(p_pa)-1, 200).astype(int)
                p_pa = p_pa[idx]
                t_k = t_k[idx]
                
            np.savetxt(
                init_pt_file, 
                np.column_stack((p_pa, t_k)), 
                fmt="%.6e",
                header="pressure temperature\nPa K",
                comments=""
            )
            return True
            
        return False

    def run(self) -> dict:
        """
        Executes the main mass-matching solver loop to find a physically 
        consistent atmosphere-interior structure.
        
        Returns:
            dict: The final model data payload.
        """
        # ==========================================
        # 0. INITIALIZATION & DYNAMIC GRID SETUP
        # ==========================================
        self.params.setdefault('T_int', 500.0) 
        
        if 'g_1bar' not in self.params:
            # First try to find a smart prior from previous runs
            smart_prior_success = self._find_closest_prior_profile(self.tmp_dir / "init_pt.dat")
            
            if smart_prior_success:
                # If we have a nearly identical prior, the old guess is perfectly safe
                self.params['g_1bar'] = self._guess_initial_gravity()
            else:
                # If this is a cold start, run the interior pre-conditioner to protect ExoREM!
                self.params['g_1bar'] = self._bootstrap_gravity()
        
        target_mass_kg = self.params['mass'] * M_JUPITER
        atm_out = None 
        
        self.p_bottom_bar = self.config.get("p_bottom_bar", 1000.0)
        p_bottom_pa = self.p_bottom_bar * 1e5
        
        if 'atmosphere_parameters' not in self.params:
            self.params['atmosphere_parameters'] = {}
        self.params['atmosphere_parameters']['pressure_max'] = p_bottom_pa
        
        init_pt_file = self.tmp_dir / "init_pt.dat"
        smart_prior_success = self._find_closest_prior_profile(init_pt_file)
        
        if not smart_prior_success:
            p_grid = np.logspace(-1, np.log10(p_bottom_pa), 81)
            t_irr = self.params.get('T_irr', 500.0)
            t_top = max(t_irr * 0.6, 200.0)
            t_grid = np.zeros_like(p_grid)
            
            for i, p in enumerate(p_grid):
                if p <= 1e4:  
                    t_grid[i] = t_top
                else:         
                    t_grid[i] = t_top * (p / 1e4) ** 0.286
                    
            t_grid = np.maximum(t_grid, self.params.get('T_int') * 0.5)
            
            np.savetxt(
                init_pt_file, 
                np.column_stack((p_grid, t_grid)), 
                fmt="%.6e",
                header="pressure temperature\nPa K",
                comments=""
            )
            logging.info(
                f"🌌 Grid Setup: Generated mathematical cold-start prior "
                f"down to {self.p_bottom_bar} bars."
            )
        
        path_str = str(self.tmp_dir)
        if not path_str.endswith('/'):
            path_str += '/'
            
        if 'paths' not in self.params:
            self.params['paths'] = {}
        self.params['paths']['path_temperature_profile'] = path_str
        
        if 'retrieval_parameters' not in self.params:
            self.params['retrieval_parameters'] = {}
        self.params['retrieval_parameters']['temperature_profile_file'] = init_pt_file.name

        # ==========================================
        # ITERATIVE ROOT-FINDING LOOP
        # ==========================================
        for iteration in range(1, self.max_iterations + 1):
            current_g = self.params['g_1bar']
            static_t_int = self.params['T_int']
            
            logging.info(f"\n{'='*50}")
            logging.info(
                f"🔄 ITERATION {iteration}/{self.max_iterations} | "
                f"Target Mass: {self.params['mass']} M_Jup | g: {current_g:.2f} m/s²"
            )
            logging.info(f"{'='*50}")
            
            # --- Warm Start Injection ---
            if iteration > 1 and atm_out is not None:
                warm_start_file = self.tmp_dir / "warm_start_pt.dat"
                p_pa = atm_out.pressure_levels
                t_k = atm_out.temperature_levels
                
                np.savetxt(
                    warm_start_file, 
                    np.column_stack((p_pa, t_k)), 
                    fmt="%.6e",
                    header="pressure temperature\nPa K",
                    comments="" 
                )
                
                self.params['paths']['path_temperature_profile'] = path_str
                self.params['retrieval_parameters']['temperature_profile_file'] = warm_start_file.name
                logging.info(f"🔥 Warm Start: Injecting P-T profile from iteration {iteration - 1}")

            # --- A. RUN ATMOSPHERE (EXOWRAP) ---
            atm_sim = Simulation(
                params=self.params, 
                resolution=self.config.get('resolution', 50)
            )
            raw_atm_df = atm_sim.run()
            
            if raw_atm_df.empty:
                logging.error("❌ exowrap failed to return data.")
                return {'status': 'failed', 'history': self.history}
                
            atm_out = ExoremOut(raw_atm_df)
            p_link, t_link = self._extract_boundary_conditions(atm_out)
            
            # --- B. ENVELOPE COMPOSITION (Z-PROFILE) ---
            sigma_val = self.params.get('sigma_val', 0.0) 
            z_base, y_ratio = calculate_z_base(
                atm_out=atm_out, 
                p_link_bar=self.p_link_bar, 
                fallback_met=self.params.get('Met', 0.0)
            )
            
            self.params['z_base'] = z_base
            self.params['Y_ratio'] = y_ratio
            
            z_profile = np.round(generate_gaussian_z_profile(
                n_layers=100, 
                sigma=sigma_val, 
                z_base=z_base, 
                z_core=0.99
            ), 3)

            # --- C. RUN INTERIOR (FUZZYCORE) ---
            fc_params = {
                'P_surf': p_link,
                'T_surf': t_link,
                'M_core': self.params.get('core_mass_earth', 10.0) * c.M_EARTH, 
                'M_water':  self.params.get('M_water', 0.0) * c.M_EARTH,
                'iron_fraction': self.params.get('iron_fraction', 0.33),
                'z_base': z_base,
                'Y_ratio': y_ratio,                          
                'sigma_val': sigma_val,
                'z_profile': z_profile,
                'initial_log_pc': 9.0,                      
                'debug': self.params.get('debug', False)     
            }
            
            int_results = solve_structure(
                target_val=current_g,
                params=fc_params,
                mode='gravity',                              
                trial_id=f"iter_{iteration}",
                csv_file=str(self.csv_path),
                write_lock=DummyLock()
            )
            
            if int_results is None:
                logging.error("❌ fuzzycore solver failed to converge on an internal structure.")
                return {
                    'status': 'failed', 
                    'history': self.history, 
                    'atmosphere_raw': raw_atm_df
                }
            
            # --- D. CALCULATE STATE ERRORS & MASS ---
            new_mass_kg, interior_mass_kg, m_atm_kg = calculate_stitched_mass(
                atm_out, int_results, self.p_link_bar
            )
            
            mass_error = (new_mass_kg - target_mass_kg) / target_mass_kg
            true_t_int = calculate_new_tint(atm_out, fallback_t_int=static_t_int)
            
            self.history['iteration'].append(iteration)
            self.history['mass_error'].append(mass_error)
            self.history['T_int_measured'].append(true_t_int)
            self.history['g_1bar'].append(current_g)
            self.history['mass_calculated'].append(new_mass_kg / M_JUPITER)
            
            logging.info(f"📊 Breakdown: Interior Mass = {interior_mass_kg / M_JUPITER:.4f} M_Jup")
            logging.info(f"📊 Breakdown: Atm Mass = {m_atm_kg / M_JUPITER:.6f} M_Jup ({m_atm_kg/new_mass_kg:.3%} of total)")
            logging.info(f"📊 Results: Total Calc Mass = {new_mass_kg / M_JUPITER:.3f} M_Jup (Error: {mass_error:.2%})")
            logging.info(f"📊 Results: True Measured T_int = {true_t_int:.1f} K (Input dial: {static_t_int} K)")

            # --- E. HIGH-RES UPGRADE, PHOTOMETRY & STITCHING ---
            output_params = self.params.copy()
            output_params['T_int_input_dial'] = static_t_int
            output_params['T_int'] = true_t_int 
            output_params['true_mass_Mjup'] = new_mass_kg / M_JUPITER
            output_params['p_link_bar'] = self.p_link_bar
            
            target_res = self.config.get('target_resolution', None)
            current_res = self.config.get('resolution', 50)
            
            if target_res and target_res > current_res:
                logging.info(f"✨ Upgrading atmosphere to R={target_res} for iteration {iteration}...")
                
                unique_hash = uuid.uuid4().hex[:8]
                unique_tmp_dir = self.output_dir / f"high_res_tmp_{unique_hash}"
                
                try:
                    raw_atm_df_high_res = upgrade_resolution(
                        results=atm_out,
                        base_params=output_params,
                        target_resolution=target_res,
                        output_dir=str(unique_tmp_dir)
                    )
                    
                    atm_out = ExoremOut(raw_atm_df_high_res)
                    raw_atm_df = raw_atm_df_high_res  
                    logging.info("🌟 High-resolution upgrade seamlessly injected!")
                    
                except Exception as e:
                    logging.error(f"⚠️ High-resolution upgrade failed ({e}). Keeping low-res model.")
                    
                finally:
                    if unique_tmp_dir.exists():
                        shutil.rmtree(unique_tmp_dir, ignore_errors=True)

            step_profile = build_master_profile(atm_out, int_results, p_link)
            photometry_section = calculate_comprehensive_photometry(atm_out)

            step_data = {
                'status': 'intermediate',
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration,
                'parameters': output_params,
                'mass_calculated_mjup': new_mass_kg / M_JUPITER,
                'mass_error': mass_error,
                'profile': step_profile,
                'atmosphere_raw': raw_atm_df,
                'interior_raw': int_results,
                'photometry': photometry_section
            }
            save_step_model(step_data, self.output_dir)
            
            # --- F. CHECK CONVERGENCE & SAVE FINAL ---
            if abs(mass_error) < self.mass_tol:
                logging.info(f"✅ CONVERGED in {iteration} iterations!")
                
                converged_results = {
                    'status': 'converged',
                    'iterations': iteration,
                    'final_params': output_params,
                    'stitched_profile': step_profile, 
                    'atmosphere_raw': raw_atm_df,  
                    'interior_raw': int_results,
                    'photometry': photometry_section
                }
                
                save_converged_model(converged_results, self.output_dir)
                return converged_results
            
            # --- G. SECANT METHOD UPDATES (GRAVITY ADJUSTMENT) ---
            if iteration == 1:
                correction = max(min(mass_error, 0.05), -0.05) 
                next_g = current_g * (1.0 + correction)
                logging.info("📈 Secant Prep: Nudging gravity to establish mass gradient.")
            else:
                g_n = current_g
                g_n_minus_1 = self.history['g_1bar'][-2]
                e_n = mass_error
                e_n_minus_1 = self.history['mass_error'][-2]
                
                if e_n == e_n_minus_1:
                    next_g = g_n * 1.01 
                    logging.warning("⚠️ Gradient stalled. Applying manual 1% push.")
                else:
                    next_g = g_n - e_n * ((g_n - g_n_minus_1) / (e_n - e_n_minus_1))
                    logging.info("🎯 Secant Method applied to Mass target.")
                    
                # Relative dampening to prevent wild percentage swings
                next_g = max(min(next_g, g_n * 1.5), g_n * 0.5) 

            # --- NEW: ABSOLUTE PHYSICAL FLOOR ---
            # Protect ExoREM's Fortran backend from dividing by zero or 
            # exploding the atmospheric scale height on runaway secant extrapolations.
            if next_g < 1.5:
                logging.warning(f"⚠️ Secant solver requested dangerously low g = {next_g:.3f} m/s²!")
                logging.warning("   Clamping to absolute minimum of 1.0 m/s² to protect ExoREM.")
                next_g = 1.5

            self.params['g_1bar'] = next_g

        # ==========================================
        # FAILURE HANDLING
        # ==========================================
        logging.warning(
            f"❌ Reached maximum iterations ({self.max_iterations}) "
            "without convergence."
        )
        
        fail_results = {
            'status': 'failed', 
            'history': self.history,
            'atmosphere_raw': getattr(atm_out, 'df', None)  
        }
        
        save_failed_run(
            self.history, 
            self.params, 
            "Max iterations reached without convergence.", 
            self.output_dir
        )
        return fail_results