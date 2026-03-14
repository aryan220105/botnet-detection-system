"""
Master launcher for the AI-Powered Botnet Detection System.

Orchestrates the full pipeline:
  1. Train ML models (if not already present)
  2. Start the detection engine
  3. Launch the Flask SOC dashboard

Usage
-----
    python scripts/run_system.py              # full pipeline
    python scripts/run_system.py --skip-train # skip training, use existing models
    python scripts/run_system.py --port 8080  # custom dashboard port
"""

import os
import sys
import argparse

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import log_system


def check_models_exist() -> bool:
    """Return True if both trained models are already on disk."""
    model_dir = os.path.join(PROJECT_ROOT, "models")
    rf = os.path.exists(os.path.join(model_dir, "random_forest_model.pkl"))
    iso = os.path.exists(os.path.join(model_dir, "isolation_forest_model.pkl"))
    return rf and iso


def train_models():
    """Run the model-training pipeline."""
    print("\n" + "=" * 60)
    print("  STEP 1 - Training ML models")
    print("=" * 60)
    from models.train_model import main as train_main
    train_main()


def launch_dashboard(host: str, port: int):
    """Start the Flask dashboard (blocking call)."""
    print("\n" + "=" * 60)
    print(f"  STEP 2 - Launching SOC Dashboard on http://{host}:{port}")
    print("=" * 60)
    from dashboard.app import run_dashboard
    run_dashboard(host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Botnet Detection System - Launcher"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model training and use existing .pkl files",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Dashboard bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Dashboard port (default: 5000)",
    )
    args = parser.parse_args()

    print("""
    ===========================================================
    |     AI-Powered Botnet Detection System                  |
    |     Network Security Mini Project                       |
    ===========================================================
    """)

    log_system("System startup initiated.")

    if args.skip_train and check_models_exist():
        print("[*] Skipping training - existing models found.")
        log_system("Skipping training (--skip-train).")
    elif check_models_exist() and args.skip_train:
        print("[*] Models already trained. Use --skip-train to skip.")
    else:
        if not check_models_exist():
            print("[*] No trained models found. Training now...")
        train_models()

    launch_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
