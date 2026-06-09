import os
import subprocess
import sys
import argparse


def ensure_saved_weights_dir(path="saved_weights"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def run_command(cmd: list) -> int:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Prepare and train models if saved weights are missing")
    # MLP hyperparams
    parser.add_argument("--mlp-num-trials", type=int, default=10)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--mlp-lr", type=float, default=0.001)
    parser.add_argument("--save-dir", type=str, default="saved_weights")
    # XGBoost hyperparams
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--xgb-lr", type=float, default=0.1)

    args = parser.parse_args()

    ensure_saved_weights_dir(args.save_dir)

    mlp_path = os.path.join(args.save_dir, "best_baseline_mlp.pth")
    xgb_path = os.path.join(args.save_dir, "best_xgboost.json")

    # Train MLP if missing
    if not exists(mlp_path):
        print("MLP weights not found, training MLP...")
        cmd = [
            sys.executable,
            "train_baseline_mlp.py",
            f"--num-trials", str(args.mlp_num_trials),
            f"--epochs", str(args.mlp_epochs),
            f"--lr", str(args.mlp_lr),
            f"--save-dir", args.save_dir,
        ]
        rc = run_command(cmd)
        if rc != 0:
            print("Warning: training MLP failed (exit code", rc, ").")
        else:
            print("MLP training finished.")
    else:
        print("MLP weights found:", mlp_path)

    # Train XGBoost if missing
    if not exists(xgb_path):
        print("XGBoost model not found, training XGBoost...")
        cmd = [
            sys.executable,
            "train_xgboost.py",
            f"--n-estimators", str(args.xgb_n_estimators),
            f"--max-depth", str(args.xgb_max_depth),
            f"--learning-rate", str(args.xgb_lr),
        ]
        rc = run_command(cmd)
        if rc != 0:
            print("Warning: training XGBoost failed (exit code", rc, ").")
        else:
            print("XGBoost training finished.")
    else:
        print("XGBoost model found:", xgb_path)

    print("Prepare step completed. Check saved_weights/ for trained models.")


if __name__ == "__main__":
    main()