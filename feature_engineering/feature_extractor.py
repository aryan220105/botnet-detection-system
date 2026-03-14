"""
Feature Extraction Module for Botnet Detection System.

Converts raw packet data into numerical features suitable for
machine-learning classification.  Operates on both live-captured
packets (Scapy packet objects) and pre-loaded pandas DataFrames.
"""

import math
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# Protocol number → human-readable name (subset)
PROTOCOL_MAP = {1: "ICMP", 6: "TCP", 17: "UDP"}


# ──────────────────────────────────────────────
# DNS entropy helper
# ──────────────────────────────────────────────

def calculate_dns_entropy(domain: str) -> float:
    """
    Shannon entropy of a domain name string.

    High entropy often indicates algorithmically-generated domains
    (DGA) used by botnets for C&C communication.
    """
    if not domain:
        return 0.0
    freq: dict[str, int] = defaultdict(int)
    for ch in domain:
        freq[ch] += 1
    length = len(domain)
    return -sum(
        (count / length) * math.log2(count / length) for count in freq.values()
    )


# ──────────────────────────────────────────────
# Per-packet feature extraction (live capture)
# ──────────────────────────────────────────────

def extract_packet_features(packet) -> Optional[dict]:
    """
    Extract ML-ready features from a single Scapy packet object.

    Returns None for packets that lack an IP layer (e.g. pure ARP).
    """
    try:
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.dns import DNS, DNSQR

        if not packet.haslayer(IP):
            return None

        ip_layer = packet[IP]
        features: dict = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": ip_layer.src,
            "destination_ip": ip_layer.dst,
            "protocol_type": ip_layer.proto,
            "packet_size": len(packet),
            "ttl": ip_layer.ttl,
            "src_port": 0,
            "dst_port": 0,
            "dns_entropy": 0.0,
            "dns_query": "",
        }

        if packet.haslayer(TCP):
            features["src_port"] = packet[TCP].sport
            features["dst_port"] = packet[TCP].dport
        elif packet.haslayer(UDP):
            features["src_port"] = packet[UDP].sport
            features["dst_port"] = packet[UDP].dport

        if packet.haslayer(DNS) and packet.haslayer(DNSQR):
            qname = packet[DNSQR].qname.decode(errors="ignore").rstrip(".")
            features["dns_query"] = qname
            features["dns_entropy"] = calculate_dns_entropy(qname)

        return features

    except Exception:
        return None


# ──────────────────────────────────────────────
# Flow-level aggregation
# ──────────────────────────────────────────────

class FlowAggregator:
    """
    Aggregates individual packets into bidirectional flows and
    computes flow-level statistics used as ML features.

    A flow is keyed by (src_ip, dst_ip, protocol).
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.flows: dict[tuple, list] = defaultdict(list)

    def add_packet(self, features: dict) -> None:
        key = (
            features["source_ip"],
            features["destination_ip"],
            features["protocol_type"],
        )
        self.flows[key].append(features)

    def compute_flow_features(self) -> pd.DataFrame:
        """Return a DataFrame with one row per flow."""
        records = []
        for (src, dst, proto), packets in self.flows.items():
            sizes = [p["packet_size"] for p in packets]
            timestamps = [p["timestamp"] for p in packets]

            t_start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            t_end = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
            duration = max((t_end - t_start).total_seconds(), 0.001)

            dns_entropies = [p["dns_entropy"] for p in packets if p["dns_entropy"] > 0]

            records.append(
                {
                    "source_ip": src,
                    "destination_ip": dst,
                    "protocol_type": proto,
                    "packet_size": np.mean(sizes),
                    "flow_duration": duration,
                    "packets_per_flow": len(packets),
                    "bytes_per_flow": sum(sizes),
                    "connection_frequency": len(packets) / duration,
                    "dns_entropy": np.mean(dns_entropies) if dns_entropies else 0.0,
                    "src_port": packets[0].get("src_port", 0),
                    "dst_port": packets[0].get("dst_port", 0),
                }
            )

        return pd.DataFrame(records)

    def reset(self) -> None:
        self.flows.clear()


# ──────────────────────────────────────────────
# DataFrame-level feature engineering
# ──────────────────────────────────────────────

FEATURE_COLUMNS = [
    "packet_size",
    "flow_duration",
    "protocol_type",
    "packets_per_flow",
    "bytes_per_flow",
    "connection_frequency",
    "dns_entropy",
    "src_port",
    "dst_port",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean the feature columns expected by the ML model.

    Missing values are filled with 0; infinite values are replaced
    with the column max.
    """
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    features = df[FEATURE_COLUMNS].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    return features
