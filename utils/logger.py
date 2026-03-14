"""
Logging utility for the Botnet Detection System.

Provides centralized logging for alerts, traffic events, and system diagnostics.
Logs are written to both console and rotating log files under the logs/ directory.
"""

import logging
import os
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ALERT_LOG = os.path.join(LOG_DIR, "alerts.log")
TRAFFIC_LOG = os.path.join(LOG_DIR, "traffic.log")
SYSTEM_LOG = os.path.join(LOG_DIR, "system.log")
ALERT_JSON = os.path.join(LOG_DIR, "alerts.json")


def _create_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Create a named logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


system_logger = _create_logger("system", SYSTEM_LOG)
traffic_logger = _create_logger("traffic", TRAFFIC_LOG)
alert_logger = _create_logger("alert", ALERT_LOG, level=logging.WARNING)


def log_alert(
    source_ip: str,
    dest_ip: str,
    threat_level: str,
    attack_type: str,
    details: str = "",
) -> dict:
    """
    Log a security alert when botnet or suspicious activity is detected.

    Returns the alert record as a dictionary so it can also be stored
    in the in-memory alert list used by the dashboard.
    """
    alert_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": source_ip,
        "destination_ip": dest_ip,
        "threat_level": threat_level,
        "attack_type": attack_type,
        "details": details,
    }

    alert_logger.warning(
        "ALERT | src=%s dst=%s threat=%s type=%s | %s",
        source_ip,
        dest_ip,
        threat_level,
        attack_type,
        details,
    )

    try:
        existing: list = []
        if os.path.exists(ALERT_JSON):
            with open(ALERT_JSON, "r") as f:
                existing = json.load(f)
        existing.append(alert_record)
        # Keep last 10 000 alerts to prevent unbounded growth
        existing = existing[-10_000:]
        with open(ALERT_JSON, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as exc:
        system_logger.error("Failed to write alert JSON: %s", exc)

    return alert_record


def log_traffic(message: str) -> None:
    """Log a general traffic observation."""
    traffic_logger.info(message)


def log_system(message: str, level: str = "info") -> None:
    """Log a system-level event (startup, shutdown, errors)."""
    getattr(system_logger, level, system_logger.info)(message)


def get_recent_alerts(n: int = 100) -> list:
    """Return the *n* most recent alerts from the JSON log."""
    try:
        if os.path.exists(ALERT_JSON):
            with open(ALERT_JSON, "r") as f:
                alerts = json.load(f)
            return alerts[-n:]
    except Exception:
        pass
    return []
