import logging
import numpy as np

# External dependencies
from exowrap.model import Simulation
from exowrap.output import ExoremOut
from fuzzycore.solver import solve_structure
from fuzzycore.utils import DummyLock, generate_gaussian_z_profile
from fuzzycore.constants import G_CONST, M_JUPITER, R_JUPITER

# Internal exolinker modules
from .profile import build_master_profile
from .physics import calculate_new_tint

class ExoCoupler:
    """
    Orchestrates the iterative coupling between exowrap (Atmosphere) 
    and fuzzycore (Interior) models using advanced numerical root-finding.
    """
    def __init__(self, target_params: dict, config: dict):
        self.params = target_params.copy()
        self.config = config
        
        # Convergence Settings
        self.max_iterations = config.get("max_iterations", 15)
        self.mass_tol = config.get("mass_convergence_threshold", 0.01)
        self.t_int_tol = config.get("t_int_convergence_threshold", 0.01)
        self.p_link_bar = config.get("p_link_target_bar", 100.0)
        
        # State tracking for the Secant Method and debugging
        self.history = {
            'iteration': [], 
            'T_int': [], 
            'g_1bar': [], 
            'mass_calculated': [], 
            'mass_error': [],
            't_int_error': []
        }

    def _guess_initial_gravity(self) -> float:
        """Calculates a smart initial gravity guess based on a basic Mass-Radius empirical prior."""
        mass_kg = self.params['mass'] * M_JUPITER
        radius_guess = R_JUPITER 
        if self.params['mass'] < 0.5:
            radius_guess = R_JUPITER * (self.params['mass'] ** 0.3)
            
        initial_g = (G_CONST * mass_kg) / (radius_guess ** 2)
        logging.info(f"💡 Smart Initialization: Guessed g = {initial_g:.2f} m/s² for M = {self.params['mass']} M_Jup")
        return float(initial_g)

    def _extract_boundary_conditions(self, atm_out: ExoremOut) -> tuple:
        """Finds the precise pressure and temperature from the atmosphere to link the models."""
        p_levels_bar = atm_out.pressure_levels / 1e5
        t_levels = atm_out.temperature_levels
        
        idx_link = np.argmin(np.abs(p_levels_bar - self.p_link_bar))
        return float(p_levels_bar[idx_link]), float(t_levels[idx_link])

    def run(self) -> dict:
        # ==========================================
        # 1. INITIALIZATION
        # ==========================================
        self.params.setdefault('T_int', 500.0)
        if 'g_1bar' not in self.params:
            self.params['g_1bar'] = self._guess_initial_gravity()
        
        target_mass_kg = self.params['mass'] * M_JUPITER
        
        # ==========================================
        # 2. ITERATIVE LOOP
        # ==========================================
        for iteration in range(1, self.max_iterations + 1):
            current_g = self.params['g_1bar']
            current_t_int = self.params['T_int']
            
            logging.info(f"\n{'='*40}")
            logging.info(f"🔄 ITERATION {iteration}/{self.max_iterations} | T_int: {current_t_int:.1f} K | g: {current_g:.2f} m/s²")
            logging.info(f"{'='*40}")
            
            # --- A. RUN ATMOSPHERE (exowrap) ---
            atm_sim = Simulation(params=self.params, resolution=self.config.get('resolution', 50))
            raw_atm_df = atm_sim.run()
            
            if raw_atm_df.empty:
                logging.error("❌ exowrap failed to return data.")
                return {'status': 'failed', 'history': self.history}
                
            atm_out = ExoremOut(raw_atm_df)
            p_link, t_link = self._extract_boundary_conditions(atm_out)
            
            # --- B. ENVELOPE COMPOSITION (Z-Profile) ---
            sigma_val = self.params.get('sigma_val', 0.0) 
            z_base = self.params.get('z_base', 0.01)      
            
            z_profile = np.round(generate_gaussian_z_profile(
                n_layers=100, 
                sigma=sigma_val, 
                z_base=z_base, 
                z_core=0.99
            ), 3)

            # --- C. RUN INTERIOR (fuzzycore) ---
            fc_params = {
                'P_surf': p_link,
                'T_surf': t_link,
                'M_core': self.params.get('core_mass_earth', 10.0) * 5.972e24, 
                'iron_fraction': self.params.get('iron_fraction', 0.33),
                'z_base': z_base,
                'sigma_val': sigma_val,
                'z_profile': z_profile,
                'initial_log_pc': 12.5,                      
                'debug': self.params.get('debug', False)     
            }
            
            # THE FIX: Anchor the interior to the exact gravity the atmosphere just used!
            int_results = solve_structure(
                target_val=current_g,
                params=fc_params,
                mode='gravity',                              
                trial_id=f"iter_{iteration}",
                csv_file="solver_steps.csv",
                write_lock=DummyLock()
            )
            
            if int_results is None:
                logging.error("❌ fuzzycore solver failed to converge on an internal structure.")
                return {
                    'status': 'failed', 
                    'history': self.history,
                    'atmosphere_raw': raw_atm_df  
                }
            
            # --- D. CALCULATE STATE ERRORS ---
            # Because gravity is locked, the error is now in the resulting Total Mass
            new_mass_kg = int_results['M'][-1]
            mass_error = (new_mass_kg - target_mass_kg) / target_mass_kg
            
            new_t_int = calculate_new_tint(atm_out, fallback_t_int=current_t_int)
            t_int_error = abs(new_t_int - current_t_int) / current_t_int
            
            self.history['iteration'].append(iteration)
            self.history['mass_error'].append(mass_error)
            self.history['t_int_error'].append(t_int_error)
            self.history['T_int'].append(new_t_int)
            self.history['g_1bar'].append(current_g)
            self.history['mass_calculated'].append(new_mass_kg / M_JUPITER)
            
            logging.info(f"📊 Results: Calc Mass = {new_mass_kg / M_JUPITER:.3f} M_Jup (Error: {mass_error:.2%})")
            logging.info(f"📊 Results: T_int = {new_t_int:.1f} K (Error: {t_int_error:.2%})")
            
            # --- E. CHECK CONVERGENCE ---
            if abs(mass_error) < self.mass_tol and t_int_error < self.t_int_tol:
                logging.info(f"✅ CONVERGED in {iteration} iterations!")
                final_profile = build_master_profile(atm_out, int_results, p_link)
                
                return {
                    'status': 'converged',
                    'iterations': iteration,
                    'final_params': self.params,
                    'stitched_profile': final_profile,
                    'atmosphere_raw': raw_atm_df,
                    'interior_raw': int_results
                }
                
            # --- F. SECANT METHOD UPDATES ---
            
            # 1. Update T_int using heavy damping (70% old, 30% new) to prevent oscillations
            self.params['T_int'] = (current_t_int * 0.7) + (new_t_int * 0.3)
            
            # 2. Update Gravity via the Secant Method targeting Mass Error
            if iteration == 1:
                # If mass_error > 0 (too heavy), we must INCREASE gravity to compress the planet's radius
                correction = max(min(mass_error, 0.05), -0.05) 
                next_g = current_g * (1.0 + correction)
                logging.info(f"📈 Secant Prep: Nudging gravity to establish mass gradient.")
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
                    
                next_g = max(min(next_g, g_n * 1.5), g_n * 0.5) 

            self.params['g_1bar'] = next_g

        logging.warning(f"❌ Reached maximum iterations ({self.max_iterations}) without convergence.")
        return {
            'status': 'failed', 
            'history': self.history,
            'atmosphere_raw': getattr(atm_out, 'df', None)  
        }