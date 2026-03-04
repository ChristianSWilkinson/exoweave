import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import os

# Resolve the absolute path to the exoweave project root
EXOWEAVE_ROOT = Path(__file__).resolve().parents[2]

# External dependencies
from exowrap.model import Simulation
from exowrap.output import ExoremOut
from fuzzycore.solver import solve_structure
from fuzzycore.utils import DummyLock, generate_gaussian_z_profile
from fuzzycore.constants import G_CONST, M_JUPITER, R_JUPITER

# Internal exoweave modules
from .profile import build_master_profile
from .physics import calculate_new_tint, calculate_z_base, calculate_stitched_mass
from .io import save_step_model, save_converged_model, save_failed_run

class ExoCoupler:
    """
    Orchestrates the iterative coupling between exowrap (Atmosphere) 
    and fuzzycore (Interior) models using pure 1D mass-targeting root-finding.
    """
    def __init__(self, target_params: dict, config: dict):
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
        # ------------------------------

        # Convergence Settings
        self.max_iterations = config.get("max_iterations", 15)
        self.mass_tol = config.get("mass_convergence_threshold", 0.01)
        self.p_link_bar = config.get("p_link_target_bar", 100.0)
        self.p_bottom_bar = config.get("p_bottom_bar", 1000.0) 
        
        self.history = {
            'iteration': [], 
            'T_int_measured': [], 
            'g_1bar': [], 
            'mass_calculated': [], 
            'mass_error': []
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
        """
        Dynamically locates the junction point by identifying the thickest 
        continuous convective block, gracefully ignoring numerical flickering.
        """
        p_levels_bar = atm_out.pressure_levels / 1e5
        t_levels = atm_out.temperature_levels
        
        try:
            is_conv = np.asarray(atm_out._get('/outputs/levels/is_convective')).astype(bool)
        except Exception as e:
            logging.warning(f"Could not find convective flags ({e}). Falling back to static target.")
            idx_link = np.argmin(np.abs(p_levels_bar - self.p_link_bar))
            return float(p_levels_bar[idx_link]), float(t_levels[idx_link])

        # 1. Sort top-to-bottom (Space to Deep Interior)
        sort_idx = np.argsort(p_levels_bar)
        p_sorted = p_levels_bar[sort_idx]
        t_sorted = t_levels[sort_idx]
        conv_sorted = is_conv[sort_idx]
        
        # 2. Map out every contiguous block of convection in the atmosphere
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
            logging.warning("No convective regions found! Falling back to static target.")
            idx_link = np.argmin(np.abs(p_levels_bar - self.p_link_bar))
            return float(p_levels_bar[idx_link]), float(t_levels[idx_link])

        # 3. Find the "Main Envelope" (The block with the largest log-pressure span)
        best_block = None
        max_span = -1.0
        
        for start, end in blocks:
            # Measure how many decades of pressure this block covers
            span = np.log10(p_sorted[end] / p_sorted[start])
            if span > max_span:
                max_span = span
                best_block = (start, end)

        # 4. Anchor exactly to the TOP of the main envelope block!
        top_idx = best_block[0]
        self.p_link_bar = float(p_sorted[top_idx])
        
        logging.info(f"🔗 Dynamic Junction: Anchoring to thickest convective block at P = {self.p_link_bar:.2f} bar")
        return float(p_sorted[top_idx]), float(t_sorted[top_idx])
    
    def _find_closest_prior_profile(self, init_pt_file: Path) -> bool:
        """
        Scans the output directory for previously saved models to use as a 
        warm-started prior, drastically reducing iteration time.
        """
        import pickle
        
        best_file = None
        best_distance = float('inf')
        threshold = 0.30  # Max 30% aggregate parameter difference allowed
        
        # Scan all .pkl files in the output directory and subdirectories
        for pkl_path in self.output_dir.glob("**/*.pkl"):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    
                if 'parameters' not in data or 'profile' not in data:
                    continue
                    
                saved = data['parameters']
                
                # Calculate normalized Euclidean distance between parameters
                m_diff = abs(saved.get('mass', 1.0) - self.params['mass']) / self.params['mass']
                t_int_diff = abs(saved.get('T_int', 500) - self.params['T_int']) / max(self.params['T_int'], 1)
                t_irr_diff = abs(saved.get('T_irr', 500) - self.params.get('T_irr', 500)) / max(self.params.get('T_irr', 500), 1)
                met_diff = abs(saved.get('Met', 0.0) - self.params.get('Met', 0.0))
                
                dist = np.sqrt(m_diff**2 + t_int_diff**2 + t_irr_diff**2 + met_diff**2)
                
                if dist < best_distance and dist < threshold:
                    best_distance = dist
                    best_file = pkl_path
                    best_data = data
                    
            except Exception:
                continue
                
        if best_file is not None:
            logging.info(f"🧠 Smart Prior: Found neighbor model ({best_file.name}) with parameter distance {best_distance:.2f}")
            df = best_data['profile']
            p_pa = df['Pressure_bar'].values * 1e5
            t_k = df['Temperature_K'].values
            
            # SAFEGUARD: Fortran requires the file bounds to fully cover its internal grid.
            # We must ensure the file goes from 0.1 Pa down to the target bottom pressure!
            target_p_bottom = self.p_bottom_bar * 1e5
            
            if p_pa[0] > 0.1:
                p_pa = np.insert(p_pa, 0, 0.1)
                t_k = np.insert(t_k, 0, t_k[0]) # Isothermal extension to space
                
            if p_pa[-1] < target_p_bottom:
                p_pa = np.append(p_pa, target_p_bottom)
                # Adiabatic extension to the deep interior
                t_bottom = t_k[-1] * (target_p_bottom / p_pa[-2]) ** 0.286
                t_k = np.append(t_k, t_bottom)
                
            # Downsample to ~200 points to keep the Fortran file clean and fast
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
        # ==========================================
        # 1. INITIALIZATION & DYNAMIC GRID SETUP
        # ==========================================
        self.params.setdefault('T_int', 500.0) 
        
        if 'g_1bar' not in self.params:
            self.params['g_1bar'] = self._guess_initial_gravity()
        
        target_mass_kg = self.params['mass'] * M_JUPITER
        atm_out = None 
        
        self.p_bottom_bar = self.config.get("p_bottom_bar", 1000.0)
        p_bottom_pa = self.p_bottom_bar * 1e5
        self.params.setdefault('atmosphere_parameters', {})['pressure_max'] = p_bottom_pa
        
        init_pt_file = self.tmp_dir / "init_pt.dat"
        
        # --- NEW: Try the Smart Scanner First! ---
        smart_prior_success = self._find_closest_prior_profile(init_pt_file)
        
        # --- Fallback: Mathematical Generator ---
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
            logging.info(f"🌌 Grid Setup: Generated mathematical cold-start prior down to {self.p_bottom_bar} bars.")
        
        # Inject the chosen prior into the Fortran namelist
        path_str = str(self.tmp_dir)
        if not path_str.endswith('/'):
            path_str += '/'
            
        self.params.setdefault('paths', {})['path_temperature_profile'] = path_str
        self.params.setdefault('retrieval_parameters', {})['temperature_profile_file'] = init_pt_file.name

        # ==========================================
        # 2. ITERATIVE LOOP
        # ==========================================
        for iteration in range(1, self.max_iterations + 1):
            current_g = self.params['g_1bar']
            static_t_int = self.params['T_int']
            
            logging.info(f"\n{'='*40}")
            logging.info(f"🔄 ITERATION {iteration}/{self.max_iterations} | Target Mass: {self.params['mass']} M_Jup | g: {current_g:.2f} m/s²")
            logging.info(f"{'='*40}")
            
            # --- 0. WARM START INJECTION ---
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

            # --- A. RUN ATMOSPHERE (exowrap) ---
            atm_sim = Simulation(params=self.params, resolution=self.config.get('resolution', 50))
            raw_atm_df = atm_sim.run()
            
            if raw_atm_df.empty:
                logging.error("❌ exowrap failed to return data.")
                return {'status': 'failed', 'history': self.history}
                
            atm_out = ExoremOut(raw_atm_df)
            p_link, t_link = self._extract_boundary_conditions(atm_out)
            
            # --- B. ENVELOPE COMPOSITION (Z-Profile & Y-Ratio) ---
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

            # --- C. RUN INTERIOR (fuzzycore) ---
            fc_params = {
                'P_surf': p_link,
                'T_surf': t_link,
                'M_core': self.params.get('core_mass_earth', 10.0) * 5.972e24, 
                'iron_fraction': self.params.get('iron_fraction', 0.33),
                'z_base': z_base,
                'Y_ratio': y_ratio,                          # Passed cleanly to solver
                'sigma_val': sigma_val,
                'z_profile': z_profile,
                'initial_log_pc': 12.5,                      
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
                return {'status': 'failed', 'history': self.history, 'atmosphere_raw': raw_atm_df}
            
            # --- D. CALCULATE STATE ERRORS ---
            # 1. Calculate the precise total mass using spherical atmospheric integration
            new_mass_kg, interior_mass_kg, m_atm_kg = calculate_stitched_mass(
                atm_out, int_results, self.p_link_bar
            )
            
            # 2. Calculate the state errors
            mass_error = (new_mass_kg - target_mass_kg) / target_mass_kg
            true_t_int = calculate_new_tint(atm_out, fallback_t_int=static_t_int)
            
            self.history['iteration'].append(iteration)
            self.history['mass_error'].append(mass_error)
            self.history['T_int_measured'].append(true_t_int)
            self.history['g_1bar'].append(current_g)
            self.history['mass_calculated'].append(new_mass_kg / M_JUPITER)
            
            # 3. Log the results
            logging.info(f"📊 Breakdown: Interior Mass = {interior_mass_kg / M_JUPITER:.4f} M_Jup")
            logging.info(f"📊 Breakdown: Atm Mass = {m_atm_kg / M_JUPITER:.6f} M_Jup ({m_atm_kg/new_mass_kg:.3%} of total)")
            logging.info(f"📊 Results: Total Calc Mass = {new_mass_kg / M_JUPITER:.3f} M_Jup (Error: {mass_error:.2%})")
            logging.info(f"📊 Results: True Measured T_int = {true_t_int:.1f} K (Input dial: {static_t_int} K)")

            # --- E. STITCH AND SAVE CURRENT STEP ---
            step_profile = build_master_profile(atm_out, int_results, p_link)
            
            # Log the true physics into the parameters specifically for the saved output
            output_params = self.params.copy()
            output_params['T_int_input_dial'] = static_t_int
            output_params['T_int'] = true_t_int 
            
            # ADD THIS LINE: Save the dynamic linking pressure!
            output_params['p_link_bar'] = self.p_link_bar

            step_data = {
                'status': 'intermediate',
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration,
                'parameters': output_params,
                'mass_calculated_mjup': new_mass_kg / M_JUPITER,
                'mass_error': mass_error,
                'profile': step_profile,
                'atmosphere_raw': raw_atm_df,
                'interior_raw': int_results
            }
            save_step_model(step_data, self.output_dir)
            
            # --- F. CHECK CONVERGENCE ---
            if abs(mass_error) < self.mass_tol:
                logging.info(f"✅ CONVERGED in {iteration} iterations!")
                
                converged_results = {
                    'status': 'converged',
                    'iterations': iteration,
                    'final_params': output_params, # Contains the swapped T_int!
                    'stitched_profile': step_profile, 
                    'atmosphere_raw': raw_atm_df,
                    'interior_raw': int_results
                }
                
                save_converged_model(converged_results, self.output_dir)
                return converged_results
                
            # --- G. SECANT METHOD UPDATES ---
            # NOTE: T_int is NOT updated. It remains completely static for stability!
            if iteration == 1:
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
        
        fail_results = {
            'status': 'failed', 
            'history': self.history,
            'atmosphere_raw': getattr(atm_out, 'df', None)  
        }
        
        save_failed_run(self.history, self.params, "Max iterations reached without convergence.", self.output_dir)
        return fail_results