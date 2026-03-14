"""
CICIDS-2017 Dataset Downloader & Preprocessor.

Downloads the real-world CICIDS-2017 network traffic dataset from
HuggingFace (originally from the Canadian Institute for Cybersecurity)
and processes it into the feature format expected by our ML models.

The Friday capture contains Botnet, DDoS, and PortScan attacks
alongside benign traffic — ideal for botnet detection training.

Usage
-----
    python data/download_dataset.py
    python data/download_dataset.py --samples 20000
"""

import os
import sys
import argparse
import urllib.request
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

FRIDAY_URL = "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/friday.csv"
RAW_FILE = os.path.join(DATA_DIR, "friday_raw.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "cicids2017_processed.csv")

# Mapping from actual CICIDS-2017 column names to our model's feature names.
# (Column names confirmed by inspecting the HuggingFace friday.csv header.)
COLUMN_MAP = {
    "Fwd Packet Length Mean": "packet_size",
    "Flow Duration": "flow_duration",
    "Protocol": "protocol_type",
    "Total Fwd Packet": "_fwd_packets",
    "Total Bwd packets": "_bwd_packets",
    "Total Length of Fwd Packet": "_fwd_bytes",
    "Total Length of Bwd Packet": "_bwd_bytes",
    "Flow Packets/s": "connection_frequency",
    "Src Port": "src_port",
    "Dst Port": "dst_port",
    "Label": "_label_raw",
}


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with a console progress bar."""
    print(f"Downloading from:\n  {url}")
    print(f"Saving to:\n  {dest}\n")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:5.1f}%  ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        else:
            mb_done = downloaded / (1024 * 1024)
            print(f"\r  Downloaded {mb_done:.1f} MB ...", end="", flush=True)

    start = time.time()
    urllib.request.urlretrieve(url, dest, reporthook=_report)
    elapsed = time.time() - start
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"\n  Done! {size_mb:.1f} MB in {elapsed:.0f}s\n")


def process_dataset(raw_path: str, output_path: str, max_samples: int = 15000) -> pd.DataFrame:
    """
    Read the raw CICIDS-2017 CSV and transform it into our model's
    feature format with a binary label (0 = normal, 1 = botnet/attack).
    """
    print("Loading raw dataset (this may take a moment for large files)...")
    # Read in chunks to handle the large file efficiently
    chunks = []
    for chunk in pd.read_csv(raw_path, chunksize=50_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Raw dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Strip whitespace from column names (CICIDS CSV has leading spaces)
    df.columns = df.columns.str.strip()
    print(f"  Labels found: {df['Label'].value_counts().to_dict()}")

    # --- Build binary label ---
    # BENIGN = 0, all attacks = 1
    df["label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)
    print(f"  Binary label distribution: {df['label'].value_counts().to_dict()}")

    # --- Map columns ---
    available_cols = set(df.columns)
    mapped = {}

    for src_col, dst_col in COLUMN_MAP.items():
        if src_col in available_cols:
            mapped[dst_col] = df[src_col]

    mapped_df = pd.DataFrame(mapped)
    mapped_df["label"] = df["label"].values

    # Compute derived features
    if "_fwd_packets" in mapped_df.columns and "_bwd_packets" in mapped_df.columns:
        mapped_df["packets_per_flow"] = (
            mapped_df["_fwd_packets"].fillna(0) + mapped_df["_bwd_packets"].fillna(0)
        )
    else:
        mapped_df["packets_per_flow"] = 0

    if "_fwd_bytes" in mapped_df.columns and "_bwd_bytes" in mapped_df.columns:
        mapped_df["bytes_per_flow"] = (
            mapped_df["_fwd_bytes"].fillna(0) + mapped_df["_bwd_bytes"].fillna(0)
        )
    else:
        mapped_df["bytes_per_flow"] = 0

    # DNS entropy: CICIDS doesn't include DNS query strings, so we derive
    # a proxy from destination port (port 53 traffic gets higher entropy value)
    mapped_df["dns_entropy"] = mapped_df["dst_port"].apply(
        lambda p: np.random.uniform(2.5, 4.5) if p == 53 else 0.0
    )

    # Drop temporary columns
    for col in ["_fwd_packets", "_bwd_packets", "_fwd_bytes", "_bwd_bytes", "_label_raw"]:
        if col in mapped_df.columns:
            mapped_df.drop(columns=[col], inplace=True)

    # --- Clean ---
    mapped_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mapped_df.dropna(inplace=True)

    # Normalize flow_duration from microseconds to seconds
    if mapped_df["flow_duration"].max() > 1_000_000:
        mapped_df["flow_duration"] = mapped_df["flow_duration"] / 1_000_000.0

    # --- Balanced sampling ---
    normal = mapped_df[mapped_df["label"] == 0]
    attack = mapped_df[mapped_df["label"] == 1]
    print(f"\n  After cleaning: {len(normal):,} normal, {len(attack):,} attack rows")

    # Keep all attack samples if they are the minority, balance with normals
    n_attack = min(len(attack), max_samples // 2)
    n_normal = min(len(normal), max_samples - n_attack)
    print(f"  Sampling {n_normal:,} normal + {n_attack:,} attack ({n_normal + n_attack:,} total)")

    sampled = pd.concat([
        normal.sample(n=n_normal, random_state=42),
        attack.sample(n=n_attack, random_state=42) if n_attack < len(attack) else attack,
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Final column order
    final_cols = [
        "packet_size", "flow_duration", "protocol_type", "packets_per_flow",
        "bytes_per_flow", "connection_frequency", "dns_entropy",
        "src_port", "dst_port", "label",
    ]
    sampled = sampled[final_cols]

    sampled.to_csv(output_path, index=False)
    print(f"\n  Processed dataset saved to: {output_path}")
    print(f"  Shape: {sampled.shape}")
    print(f"  Label distribution:\n{sampled['label'].value_counts().to_string()}")

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Download & preprocess CICIDS-2017 dataset")
    parser.add_argument("--samples", type=int, default=15000,
                        help="Total samples in the output dataset (default: 15000)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if friday_raw.csv already exists")
    args = parser.parse_args()

    print("=" * 60)
    print("  CICIDS-2017 Dataset Downloader & Preprocessor")
    print("=" * 60)
    print()

    # Step 1: Download
    if args.skip_download and os.path.exists(RAW_FILE):
        print(f"[*] Skipping download -- raw file exists at {RAW_FILE}")
    else:
        download_with_progress(FRIDAY_URL, RAW_FILE)

    # Step 2: Process
    process_dataset(RAW_FILE, OUTPUT_FILE, max_samples=args.samples)

    print("\n[OK] Dataset ready. Now retrain models with:")
    print("     python models/train_model.py --dataset data/cicids2017_processed.csv")


if __name__ == "__main__":
    main()
