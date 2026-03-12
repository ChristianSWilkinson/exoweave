# 🪐 ExoWeave

**ExoWeave** is a high-performance planetary modeling orchestrator that seamlessly couples **ExoREM** (1D atmospheric radiative-convective models) with **FuzzyCore** (deep interior thermodynamic models). 

By leveraging advanced numerical root-finding (the Secant Method) and smart empirical initializations, ExoWeave forces atmospheric and interior models into strict thermodynamic and hydrostatic agreement, generating a continuous, stitched planetary profile from the top of the atmosphere down to the solid core.

---

## ✨ Key Features
* **Automated Ecosystem Bootstrapping**: A built-in CLI (`exoweave init`) that automatically clones, updates, and compiles all Fortran and Python dependencies interactively.
* **Secant Method Convergence**: Abandons slow proportional damping for a lightning-fast Secant root-finder, converging on planetary mass targets in just 3-4 iterations.
* **Smart Initialization**: Uses empirical Mass-Radius priors to guess the initial gravity, saving massive amounts of computational time.
* **Continuous Stitched Profiles**: Automatically handles the log-log interpolation of densities across 15 orders of magnitude, returning a clean, continuous Pandas DataFrame.
* **Compositional Gradients**: Natively supports parameterized heavy-element (Z) gradients in the envelope to model fuzzy cores and realistic compressibilities.

---

## 🚀 Installation

ExoWeave is designed to manage its own complex dependencies. 

**1. Install the Orchestrator**
Clone this repository and install it in editable mode:
```bash
git clone git@github.com:ChristianSWilkinson/exoweave.git
```
or :
```bash
git clone https://github.com/ChristianSWilkinson/exoweave.git
```
```bash
cd exoweave
pip install -e .
```

**2. The Compiler & HDF5 (The Conda Route - Recommended)**
If you are using Anaconda/Miniconda, the native HDF5 wrappers (`h5fc`) are incredibly strict about which compiler they use. **You must install Conda's Fortran compiler**.

*For Linux, Windows, or Intel Macs:*
```bash
conda create -n exowrap_env -c conda-forge fortran-compiler hdf5 python=3.10

conda activate exowrap_env
```

**3. Bootstrap the Ecosystem**
Run the built-in initialization command. This will create a hidden `~/.exolinker/src/` folder, download `exowrap` and `fuzzycore`, install them, and compile the heavy Fortran backends automatically.
```bash
exoweave init
```
*(Note: If the dependencies are already installed, this command will present a clean interactive menu asking if you want to update, re-clone, or skip).*

**4. Download High-Resolution Tables (Optional)**
If you want to run high-resolution models (R=500 or R=20000), download the corresponding K-tables:
```bash
exowrap download-tables --res 500
```

---

## 💻 Quick Start

You can run ExoWeave from any Jupyter Notebook or Python script on your machine.

```python
from exoweave import ExoCoupler, save_converged_model

# 1. Define the physical parameters of the target planet
target_params = {
    "mass": 1.0,               # Planet mass in Jupiter masses
    "T_irr": 100.0,            # Irradiation temperature (K)
    "Met": 0.0,                # Atmospheric metallicity (log10 Z/Z_solar)
    "core_mass_earth": 15.0,   # Solid core mass in Earth masses
    "iron_fraction": 0.33,     # Core composition (0.33 = Earth-like)
    "f_sed": 1.0,              # Cloud sedimentation parameter
    "kzz": 8.0,                # Eddy diffusion (log10)
    "z_base": 0.01,            # 1% heavy elements in the deep envelope
    "debug": False
}

# 2. Define the numerical configuration
config = {
    "resolution": 50,                  # exowrap K-table resolution
    "max_iterations": 15,              # Maximum solver steps
    "p_link_target_bar": 100.0,        # Stitching boundary (bar)
    "g_convergence_threshold": 0.01,   # 1% mass tolerance
    "t_int_convergence_threshold": 0.01# 1% T_int tolerance
}

# 3. Initialize and Run
coupler = ExoCoupler(target_params=target_params, config=config)
results = coupler.run()

# 4. Inspect and Save
if results['status'] == 'converged':
    print(f"Converged in {results['iterations']} iterations!")
    
    # Extract the continuous Pandas DataFrame
    df = results['stitched_profile']
    
    # Save the output safely to disk
    save_converged_model(results, output_dir="./models")
else:
    print("Solver failed. Check results['history'] for diagnostics.")
```

---

## 🏗️ Architecture

ExoWeave operates via a top-down boundary condition matching loop:
1. **Atmosphere (`exowrap`)**: Runs a full radiative-convective ExoREM model using a guessed surface gravity and T_int.
2. **Boundary Extraction**: Extracts the exact Pressure, Temperature, and Composition at the linking boundary (default: 100 bar).
3. **Interior (`fuzzycore`)**: Integrates the planetary interior downwards from the linking boundary to the core, using a mixed Rock/Iron/Water/Gas Equation of State, terminating when the equations of state are satisfied.
4. **Error Calculation & Feedback**: Calculates the total resulting mass of the interior model. ExoWeave then uses the **Secant Method** to calculate the exact gravity needed for the next atmospheric run to push the mass error to zero.

---

## 📂 Output Data Structure

When you save a converged model using `save_converged_model()`, ExoWeave generates a clean `.pkl` file. When loaded, it provides a dictionary containing:
* `status`: 'converged' or 'failed'
* `parameters`: The final converged physical parameters.
* `iterations`: Total steps taken.
* `profile`: A continuous, stitched `pd.DataFrame` containing `Pressure_bar`, `Radius_m`, `Temperature_K`, `Density_kgm3`, and `Gravity_ms2`.
* `atmosphere_raw` / `interior_raw`: The raw, unadulterated outputs from the underlying solvers for deep debugging.

---
*Developed by Christian Wilkinson.*
