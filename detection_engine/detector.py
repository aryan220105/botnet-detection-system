"""
Detection Engine for Botnet Detection System.

Loads pre-trained ML models and classifies incoming traffic features
as NORMAL, SUSPICIOUS, or BOTNET.  Integrates with the IP blacklist
and alert subsystem to provide a complete detection pipeline.
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_extractor import FEATURE_COLUMNS, prepare_features
from utils.logger import log_alert, log_system, log_traffic
from utils.ip_blacklist import is_blacklisted, add_to_blacklist, geolocate_ip

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

THREAT_LABELS = {0: "NORMAL", 1: "BOTNET"}


class BotnetDetector:
    """
    Core detection engine.

    Loads a Random Forest model (primary) and an Isolation Forest model
    (secondary/anomaly) and exposes ``classify()`` for single-packet
    or batch classification.
    """

    def __init__(
        self,
        rf_model_path: Optional[str] = None,
        iso_model_path: Optional[str] = None,
    ):
        rf_model_path = rf_model_path or os.path.join(MODEL_DIR, "random_forest_model.pkl")
        iso_model_path = iso_model_path or os.path.join(MODEL_DIR, "isolation_forest_model.pkl")

        self.rf_bundle = self._load_bundle(rf_model_path, "RandomForest")
        self.iso_bundle = self._load_bundle(iso_model_path, "IsolationForest")

        self.alert_history: list[dict] = []
        self.stats = {
            "total_analysed": 0,
            "normal": 0,
            "suspicious": 0,
            "botnet": 0,
        }

    # ──────────────────────────────────────────

    @staticmethod
    def _load_bundle(path: str, name: str) -> Optional[dict]:
        if os.path.exists(path):
            bundle = joblib.load(path)
            log_system(f"{name} model loaded from {path}")
            return bundle
        log_system(f"{name} model not found at {path} — skipping", level="warning")
        return None

    # ──────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────

    def classify(self, features: dict) -> dict:
        """
        Classify a single packet/flow feature dict.

        Returns the original features augmented with:
          classification  – NORMAL | SUSPICIOUS | BOTNET
          threat_level    – LOW | MEDIUM | HIGH | CRITICAL
          attack_type     – descriptive label
          confidence      – model probability (0-1)
        """
        self.stats["total_analysed"] += 1

        result = dict(features)
        result["classification"] = "NORMAL"
        result["threat_level"] = "LOW"
        result["attack_type"] = "None"
        result["confidence"] = 0.0

        src_ip = features.get("source_ip", "0.0.0.0")
        dst_ip = features.get("destination_ip", "0.0.0.0")

        # Rule 1: instant blacklist match → CRITICAL
        if is_blacklisted(src_ip) or is_blacklisted(dst_ip):
            result["classification"] = "BOTNET"
            result["threat_level"] = "CRITICAL"
            result["attack_type"] = "Blacklisted IP Communication"
            result["confidence"] = 1.0
            self._record_alert(result)
            return result

        # Prepare feature vector
        row = pd.DataFrame([features])
        X = prepare_features(row)

        # Rule 2: Random Forest prediction
        if self.rf_bundle:
            model = self.rf_bundle["model"]
            scaler = self.rf_bundle["scaler"]
            X_scaled = pd.DataFrame(
                scaler.transform(X), columns=FEATURE_COLUMNS
            )
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(max(proba))

            if prediction == 1:
                result["classification"] = "BOTNET"
                result["threat_level"] = self._threat_level(confidence)
                result["attack_type"] = self._infer_attack(features)
                result["confidence"] = confidence
                self._record_alert(result)
                return result

        # Rule 3: Isolation Forest anomaly check
        if self.iso_bundle:
            model = self.iso_bundle["model"]
            scaler = self.iso_bundle["scaler"]
            X_scaled = pd.DataFrame(
                scaler.transform(X), columns=FEATURE_COLUMNS
            )
            anomaly_score = model.decision_function(X_scaled)[0]
            is_anomaly = model.predict(X_scaled)[0] == -1

            if is_anomaly:
                result["classification"] = "SUSPICIOUS"
                result["threat_level"] = "MEDIUM"
                result["attack_type"] = "Anomalous Traffic Pattern"
                result["confidence"] = round(min(1.0, abs(float(anomaly_score))), 3)
                self._record_alert(result)
                return result

        # Rule 4: heuristic checks on raw feature values
        heuristic = self._heuristic_check(features)
        if heuristic:
            result.update(heuristic)
            if result["classification"] != "NORMAL":
                self._record_alert(result)
            return result

        self.stats["normal"] += 1
        return result

    # ──────────────────────────────────────────

    def classify_batch(self, feature_list: list[dict]) -> list[dict]:
        """Classify a list of feature dicts and return augmented results."""
        return [self.classify(f) for f in feature_list]

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _threat_level(confidence: float) -> str:
        if confidence >= 0.90:
            return "CRITICAL"
        if confidence >= 0.75:
            return "HIGH"
        if confidence >= 0.50:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _infer_attack(features: dict) -> str:
        """Rule-based heuristic to assign a descriptive attack label."""
        dst_port = features.get("dst_port", 0)
        dns_entropy = features.get("dns_entropy", 0)
        conn_freq = features.get("connection_frequency", 0)

        if dns_entropy > 3.5:
            return "DGA Domain Communication"
        if dst_port in (6667, 6668, 6669):
            return "IRC C&C Channel"
        if dst_port == 4444:
            return "Reverse Shell / Backdoor"
        if conn_freq > 50:
            return "C&C Beaconing"
        return "Botnet Traffic"

    @staticmethod
    def _heuristic_check(features: dict) -> Optional[dict]:
        """
        Lightweight rule-based check that fires even when no ML model
        is available (fallback layer).
        """
        dns_entropy = features.get("dns_entropy", 0)
        conn_freq = features.get("connection_frequency", 0)
        dst_port = features.get("dst_port", 0)

        if dns_entropy > 4.0:
            return {
                "classification": "SUSPICIOUS",
                "threat_level": "HIGH",
                "attack_type": "High-Entropy DNS (possible DGA)",
                "confidence": 0.7,
            }
        if dst_port in (6667, 4444) and conn_freq > 30:
            return {
                "classification": "SUSPICIOUS",
                "threat_level": "MEDIUM",
                "attack_type": "Suspicious Port + High Frequency",
                "confidence": 0.6,
            }
        return None

    def _record_alert(self, result: dict) -> None:
        """Persist alert and update running stats."""
        classification = result["classification"]
        if classification == "BOTNET":
            self.stats["botnet"] += 1
        else:
            self.stats["suspicious"] += 1

        alert = log_alert(
            source_ip=result.get("source_ip", "N/A"),
            dest_ip=result.get("destination_ip", "N/A"),
            threat_level=result["threat_level"],
            attack_type=result["attack_type"],
            details=f"confidence={result['confidence']}",
        )
        self.alert_history.append(alert)

        # Auto-blacklist IPs that trigger CRITICAL alerts
        if result["threat_level"] == "CRITICAL":
            src = result.get("source_ip", "")
            if src and not src.startswith(("192.168.", "10.", "172.16.")):
                add_to_blacklist(src)

    # ──────────────────────────────────────────
    # Public accessors
    # ──────────────────────────────────────────

    def get_stats(self) -> dict:
        return dict(self.stats)

    def get_recent_alerts(self, n: int = 50) -> list[dict]:
        return self.alert_history[-n:]


# ──────────────────────────────────────────────
# CLI quick-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from packet_capture.capture import generate_simulated_packet

    detector = BotnetDetector()
    print("\nClassifying 20 simulated packets…\n")
    for _ in range(20):
        pkt = generate_simulated_packet(malicious_ratio=0.4)
        result = detector.classify(pkt)
        tag = result["classification"]
        symbol = {"NORMAL": ".", "SUSPICIOUS": "?", "BOTNET": "!"}[tag]
        print(
            f"  [{symbol}] {result.get('source_ip','?'):>15} -> "
            f"{result.get('destination_ip','?'):<15} "
            f"| {tag:<12} | {result['threat_level']:<8} | {result['attack_type']}"
        )

    print(f"\nStats: {detector.get_stats()}")
