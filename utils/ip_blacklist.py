"""
IP Blacklist and Geo-location utilities.

Maintains a local blacklist of known malicious IPs and provides
geo-location lookups via the free ip-api.com service.
"""

import os
import json
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLACKLIST_FILE = os.path.join(BASE_DIR, "data", "ip_blacklist.json")

# Well-known botnet C&C IPs (example seed list — not real threat intel)
DEFAULT_BLACKLIST = [
    "198.51.100.1",
    "203.0.113.50",
    "192.0.2.100",
    "198.51.100.23",
    "203.0.113.99",
]


def load_blacklist() -> set:
    """Load the IP blacklist from disk, creating a default file if absent."""
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, "r") as f:
            return set(json.load(f))

    save_blacklist(set(DEFAULT_BLACKLIST))
    return set(DEFAULT_BLACKLIST)


def save_blacklist(ip_set: set) -> None:
    """Persist the blacklist set to disk."""
    os.makedirs(os.path.dirname(BLACKLIST_FILE), exist_ok=True)
    with open(BLACKLIST_FILE, "w") as f:
        json.dump(sorted(ip_set), f, indent=2)


def add_to_blacklist(ip: str) -> None:
    """Add a single IP to the blacklist."""
    bl = load_blacklist()
    bl.add(ip)
    save_blacklist(bl)


def is_blacklisted(ip: str) -> bool:
    """Check whether an IP is on the blacklist."""
    return ip in load_blacklist()


def geolocate_ip(ip: str) -> dict:
    """
    Return geo-location data for an IP address using the free ip-api service.

    Falls back to an empty dict on failure to avoid crashing the pipeline.
    """
    try:
        resp = requests.get(
            f"http://ip-api.com/json/{ip}",
            timeout=3,
            params={"fields": "status,country,regionName,city,lat,lon,isp,org"},
        )
        data = resp.json()
        if data.get("status") == "success":
            return {
                "ip": ip,
                "country": data.get("country", "Unknown"),
                "region": data.get("regionName", ""),
                "city": data.get("city", ""),
                "lat": data.get("lat", 0),
                "lon": data.get("lon", 0),
                "isp": data.get("isp", ""),
                "org": data.get("org", ""),
                "lookup_time": datetime.now().isoformat(),
            }
    except Exception:
        pass
    return {"ip": ip, "country": "Unknown", "error": "Lookup failed"}
