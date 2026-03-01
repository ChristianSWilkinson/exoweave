"""
exoweave: Seamlessly couple ExoREM atmospheres with FuzzyCore interiors.
"""
import logging

__version__ = "0.1.0"

# We wrap these in a try-except block so that if fuzzycore or exowrap 
# are broken, outdated, or missing entirely, it doesn't crash the CLI.
# This ensures `exoweave init` can always run to fix the environment!
try:
    # Expose the main orchestrator
    from .coupler import ExoCoupler

    # Expose the physical profile mapping tool
    from .profile import build_master_profile

    # Expose the cross-domain physics calculations
    from .physics import calculate_new_tint, calculate_entropy_evolution

    # Expose the input/output utilities
    from .io import save_converged_model, save_failed_run

    # Define the public API of the package
    __all__ = [
        "ExoCoupler",
        "build_master_profile",
        "calculate_new_tint",
        "calculate_entropy_evolution",
        "save_converged_model",
        "save_failed_run"
    ]

except ImportError as e:
    # Fail gracefully so the CLI can still execute 'exoweave init'
    logging.debug(f"exoweave imports deferred due to missing/outdated dependencies: {e}")
    __all__ = []