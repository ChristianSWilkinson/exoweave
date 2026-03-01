"""
Command Line Interface for ExoLinker.

Handles the automated environment bootstrapping, including downloading, 
updating, and compiling the ExoWrap and FuzzyCore dependencies.
"""

import argparse
import logging
import subprocess
import sys
import shutil
from pathlib import Path

# Constants for Github Repositories
EXOWRAP_REPO = "https://github.com/ChristianSWilkinson/exowrap.git"
FUZZYCORE_REPO = "https://github.com/ChristianSWilkinson/fuzzycore.git"

# Installation Directory
USER_HOME = Path.home()
EXOLINKER_DIR = USER_HOME / ".exolinker"
SRC_DIR = EXOLINKER_DIR / "src"

def _run_subprocess(cmd: list, cwd: Path = None, err_msg: str = "Command failed."):
    """Helper to run subprocesses securely and handle errors."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {err_msg}: {e}")
        sys.exit(1)

def _handle_repo(repo_name: str, repo_url: str, repo_path: Path):
    """
    Handles cloning, updating, or recloning a Git repository interactively 
    using a clean, user-friendly terminal menu.
    """
    if repo_path.exists():
        prompt = (
            f"\n📦 The repository '{repo_name}' is already installed. What would you like to do?\n"
            f"   [u] Update      (Pull the latest changes from GitHub)\n"
            f"   [r] Re-download (Delete the folder and clone a fresh copy)\n"
            f"   [s] Skip        (Leave it exactly as it is)\n\n"
            f"   Enter your choice (u/r/s) [Default: u]: "
        )
        choice = input(prompt).strip().lower()
        
        if choice == 'r':
            print(f"🗑️ Deleting existing {repo_name} directory...")
            shutil.rmtree(repo_path)
            print(f"📥 Re-cloning {repo_name}...")
            _run_subprocess(["git", "clone", repo_url, str(repo_path)], err_msg=f"Failed to clone {repo_name}")
        elif choice == 's':
            print(f"⏭️ Skipping {repo_name} repository update.")
        else:
            # Default behavior is to update
            print(f"🔄 Pulling latest updates for {repo_name}...")
            _run_subprocess(["git", "pull"], cwd=repo_path, err_msg=f"Failed to pull {repo_name}")
    else:
        print(f"📥 Cloning {repo_name} into {repo_path}...")
        _run_subprocess(["git", "clone", repo_url, str(repo_path)], err_msg=f"Failed to clone {repo_name}")

    # Always ensure the package is pip-installed just in case the environment dropped it
    print(f"📦 Installing {repo_name} via pip...")
    _run_subprocess([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo_path, err_msg=f"Failed to pip install {repo_name}")

def setup_exolinker(args: argparse.Namespace):
    """
    Interactively manages fuzzycore and exowrap installations and initializes 
    the Fortran backend.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("--- 🚀 Initializing ExoLinker Ecosystem ---")
    
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Manage FuzzyCore
    fuzzycore_path = SRC_DIR / "fuzzycore"
    _handle_repo("fuzzycore", FUZZYCORE_REPO, fuzzycore_path)

    # 2. Manage ExoWrap
    exowrap_path = SRC_DIR / "exowrap"
    _handle_repo("exowrap", EXOWRAP_REPO, exowrap_path)

    # 3. Trigger ExoWrap's Fortran Initialization
    print("\n⚙️ Triggering ExoWrap Fortran Backend Initialization...")
    exowrap_bin = shutil.which("exowrap")
    if not exowrap_bin:
        logging.error("❌ 'exowrap' command not found. Pip install may have failed.")
        sys.exit(1)
        
    _run_subprocess([exowrap_bin, "init"], err_msg="ExoWrap Fortran compilation failed")

    print("\n✅ ExoLinker ecosystem successfully installed and configured!")
    print("   You can now import exolinker in your Python scripts.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ExoLinker CLI: Environment bootstrapping tool."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # INIT COMMAND
    subparsers.add_parser("init", help="Clone, update, install, and compile dependencies (ExoWrap & FuzzyCore).")

    args = parser.parse_args()

    if args.command == "init":
        setup_exolinker(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()