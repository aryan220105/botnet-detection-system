"""
Packet Capture Module for Botnet Detection System.

Uses Scapy to sniff live network traffic and feeds each packet
through the feature-extraction pipeline.  Includes a simulation
mode that generates synthetic packets for testing without requiring
elevated privileges.
"""

import os
import sys
import time
import random
import threading
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_extractor import (
    extract_packet_features,
    FlowAggregator,
    FEATURE_COLUMNS,
    calculate_dns_entropy,
)
from utils.logger import log_traffic, log_system


# ──────────────────────────────────────────────
# Live packet capture (requires admin/root)
# ──────────────────────────────────────────────

def start_live_capture(
    interface: Optional[str] = None,
    packet_callback: Optional[Callable] = None,
    count: int = 0,
    timeout: int = 60,
) -> list[dict]:
    """
    Capture live packets using Scapy.

    Parameters
    ----------
    interface : str, optional
        Network interface to sniff on (None = default).
    packet_callback : callable, optional
        Called with each feature dict as packets arrive.
    count : int
        Number of packets to capture (0 = unlimited until timeout).
    timeout : int
        Seconds before stopping the capture.

    Returns
    -------
    list[dict]
        List of extracted feature dictionaries.
    """
    try:
        from scapy.all import sniff
    except ImportError:
        log_system("Scapy is not installed. Run: pip install scapy", level="error")
        return []

    captured_features: list[dict] = []

    def _process(packet):
        features = extract_packet_features(packet)
        if features:
            captured_features.append(features)
            log_traffic(
                f"Captured: {features['source_ip']} -> {features['destination_ip']} "
                f"proto={features['protocol_type']} size={features['packet_size']}"
            )
            if packet_callback:
                packet_callback(features)

    log_system(f"Starting live capture on interface={interface or 'default'} "
               f"timeout={timeout}s count={count}")

    try:
        sniff(
            iface=interface,
            prn=_process,
            count=count,
            timeout=timeout,
            store=False,
        )
    except PermissionError:
        log_system(
            "Permission denied — run as Administrator/root for live capture.",
            level="error",
        )
    except Exception as exc:
        log_system(f"Capture error: {exc}", level="error")

    log_system(f"Capture finished. {len(captured_features)} packets processed.")
    return captured_features


# ──────────────────────────────────────────────
# Simulated capture (no privileges needed)
# ──────────────────────────────────────────────

# Example DGA-style domains used for botnet simulation
_DGA_DOMAINS = [
    "xk4r9z2m.evil.com",
    "q8w7e3rt.malware.net",
    "a1b2c3d4.botnet.org",
    "zz99xx88.c2server.io",
    "m4lw4r3x.darkweb.xyz",
]

_NORMAL_DOMAINS = [
    "www.google.com",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "en.wikipedia.org",
]

_NORMAL_IPS = [
    "192.168.1.10", "192.168.1.20", "10.0.0.5", "172.16.0.100",
]

_BOTNET_IPS = [
    "198.51.100.1", "203.0.113.50", "192.0.2.100", "198.51.100.23",
]


def generate_simulated_packet(malicious_ratio: float = 0.3) -> dict:
    """
    Generate a single synthetic packet feature dict.

    Roughly *malicious_ratio* of packets exhibit botnet-like patterns
    (high DNS entropy, IRC ports, C&C beaconing frequency).
    """
    is_malicious = random.random() < malicious_ratio
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_malicious:
        domain = random.choice(_DGA_DOMAINS)
        return {
            "timestamp": now,
            "source_ip": random.choice(_BOTNET_IPS),
            "destination_ip": random.choice(_NORMAL_IPS),
            "protocol_type": random.choice([6, 17]),
            "packet_size": random.randint(60, 300),
            "ttl": random.randint(32, 64),
            "src_port": random.randint(30000, 45000),
            "dst_port": random.choice([53, 6667, 8080, 4444]),
            "dns_entropy": calculate_dns_entropy(domain),
            "dns_query": domain,
            "flow_duration": round(random.uniform(0.001, 0.02), 4),
            "packets_per_flow": random.randint(1, 5),
            "bytes_per_flow": random.randint(60, 1500),
            "connection_frequency": random.randint(40, 95),
        }

    domain = random.choice(_NORMAL_DOMAINS)
    return {
        "timestamp": now,
        "source_ip": random.choice(_NORMAL_IPS),
        "destination_ip": f"93.184.{random.randint(1,254)}.{random.randint(1,254)}",
        "protocol_type": random.choice([6, 17]),
        "packet_size": random.randint(64, 1500),
        "ttl": random.randint(64, 128),
        "src_port": random.randint(49152, 65535),
        "dst_port": random.choice([80, 443, 53, 22]),
        "dns_entropy": calculate_dns_entropy(domain),
        "dns_query": domain,
        "flow_duration": round(random.uniform(0.001, 1.0), 4),
        "packets_per_flow": random.randint(1, 250),
        "bytes_per_flow": random.randint(64, 400000),
        "connection_frequency": random.randint(1, 10),
    }


def start_simulated_capture(
    num_packets: int = 200,
    delay: float = 0.05,
    packet_callback: Optional[Callable] = None,
    malicious_ratio: float = 0.3,
) -> list[dict]:
    """
    Generate a stream of simulated packet features.

    Useful for demos and testing without needing admin privileges
    or an active network.
    """
    log_system(f"Starting SIMULATED capture: {num_packets} packets, "
               f"malicious_ratio={malicious_ratio}")
    captured: list[dict] = []

    for i in range(num_packets):
        pkt = generate_simulated_packet(malicious_ratio)
        captured.append(pkt)

        log_traffic(
            f"[SIM] {pkt['source_ip']} -> {pkt['destination_ip']} "
            f"proto={pkt['protocol_type']} size={pkt['packet_size']}"
        )

        if packet_callback:
            packet_callback(pkt)

        time.sleep(delay)

    log_system(f"Simulated capture complete. {len(captured)} packets generated.")
    return captured


# ──────────────────────────────────────────────
# Convenience: capture → DataFrame
# ──────────────────────────────────────────────

def capture_to_dataframe(packets: list[dict]) -> pd.DataFrame:
    """Convert a list of packet feature dicts into a DataFrame."""
    if not packets:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame(packets)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Packet Capture Module")
    parser.add_argument("--mode", choices=["live", "sim"], default="sim",
                        help="Capture mode: live (needs admin) or sim (simulated)")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of packets to capture")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface for live capture")
    args = parser.parse_args()

    if args.mode == "live":
        packets = start_live_capture(interface=args.interface, count=args.count)
    else:
        packets = start_simulated_capture(num_packets=args.count)

    df = capture_to_dataframe(packets)
    print(f"\nCaptured {len(df)} packets")
    print(df.head(10))
