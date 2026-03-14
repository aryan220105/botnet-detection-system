"""
Flask Web Dashboard for the Botnet Detection System.

Serves a real-time SOC-style dashboard that displays:
  * Live traffic statistics
  * Detected threats & alert log
  * Suspicious IP list with geo-location
  * Interactive charts (protocol distribution, attack timeline, traffic volume)

The dashboard feeds from a background thread that continuously generates
simulated traffic (or live-captured traffic) and pushes results into
shared data structures consumed by the Jinja templates and JSON API.
"""

import os
import sys
import json
import threading
import time
from datetime import datetime
from collections import Counter, deque

from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_engine.detector import BotnetDetector
from packet_capture.capture import generate_simulated_packet
from utils.logger import log_system, get_recent_alerts
from utils.ip_blacklist import load_blacklist, geolocate_ip

# ──────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "dashboard", "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = os.urandom(24)

detector = BotnetDetector()

# Shared state (guarded by a lock for thread safety)
_lock = threading.Lock()
traffic_log: deque[dict] = deque(maxlen=500)
alert_list: deque[dict] = deque(maxlen=500)
protocol_counter: Counter = Counter()
classification_counter: Counter = Counter()
timeline_data: list[dict] = []
suspicious_ips: dict[str, dict] = {}
_capture_running = False


# ──────────────────────────────────────────────
# Background capture thread
# ──────────────────────────────────────────────

def _background_capture():
    """Continuously generate simulated packets and classify them."""
    global _capture_running
    _capture_running = True
    log_system("Background capture thread started.")

    while _capture_running:
        pkt = generate_simulated_packet(malicious_ratio=0.3)
        result = detector.classify(pkt)

        with _lock:
            traffic_log.append(result)

            proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(
                result.get("protocol_type", 0), "Other"
            )
            protocol_counter[proto_name] += 1
            classification_counter[result["classification"]] += 1

            if result["classification"] in ("BOTNET", "SUSPICIOUS"):
                alert_list.append(result)
                src = result.get("source_ip", "")
                if src:
                    if src not in suspicious_ips:
                        suspicious_ips[src] = {
                            "ip": src,
                            "count": 0,
                            "first_seen": result.get("timestamp", ""),
                            "last_seen": result.get("timestamp", ""),
                            "threat_level": result.get("threat_level", "MEDIUM"),
                        }
                    suspicious_ips[src]["count"] += 1
                    suspicious_ips[src]["last_seen"] = result.get("timestamp", "")

            timeline_data.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "classification": result["classification"],
            })
            # Keep timeline manageable
            if len(timeline_data) > 300:
                del timeline_data[:100]

        time.sleep(0.3)


def start_background_capture():
    t = threading.Thread(target=_background_capture, daemon=True)
    t.start()
    return t


# ──────────────────────────────────────────────
# Routes — Pages
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    with _lock:
        stats = detector.get_stats()
        recent_alerts = list(alert_list)[-20:]
        top_ips = sorted(
            suspicious_ips.values(), key=lambda x: x["count"], reverse=True
        )[:10]
    return render_template(
        "index.html",
        stats=stats,
        alerts=recent_alerts,
        suspicious_ips=top_ips,
        blacklist=sorted(load_blacklist()),
    )


# ──────────────────────────────────────────────
# Routes — JSON API (consumed by Chart.js on the frontend)
# ──────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    with _lock:
        return jsonify(detector.get_stats())


@app.route("/api/alerts")
def api_alerts():
    n = request.args.get("n", 50, type=int)
    with _lock:
        return jsonify(list(alert_list)[-n:])


@app.route("/api/traffic")
def api_traffic():
    n = request.args.get("n", 30, type=int)
    with _lock:
        return jsonify(list(traffic_log)[-n:])


@app.route("/api/protocol_distribution")
def api_protocol_dist():
    with _lock:
        return jsonify(dict(protocol_counter))


@app.route("/api/classification_distribution")
def api_class_dist():
    with _lock:
        return jsonify(dict(classification_counter))


@app.route("/api/timeline")
def api_timeline():
    with _lock:
        return jsonify(timeline_data[-60:])


@app.route("/api/suspicious_ips")
def api_suspicious_ips():
    with _lock:
        top = sorted(suspicious_ips.values(), key=lambda x: x["count"], reverse=True)
    return jsonify(top[:20])


@app.route("/api/geolocate/<ip>")
def api_geolocate(ip: str):
    return jsonify(geolocate_ip(ip))


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    start_background_capture()
    log_system(f"Dashboard starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    run_dashboard(debug=True)
