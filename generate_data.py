"""
generate_data.py - creates SYNTHETIC (fake) campus network traffic.

IMPORTANT: Everything here is artificial demo data. No real traffic is captured
and no real people's IP addresses are used. IPs come from private ranges
(10.x.x.x) and the documentation ranges reserved for examples
(203.0.113.x and 198.51.100.x).

Run:  python generate_data.py
Output: data/network_traffic.csv (training/demo data, has a 'label' column)
        data/sample_upload.csv  (small file without label, for the Upload page)

The 'label' column is only used to check how well the model works. It is NEVER
used as a model feature.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.config import DATA_PATH, SAMPLE_UPLOAD_PATH

# (destination_port, protocol, share_of_traffic,
#  packets_mu, packets_sigma, bytes_per_packet, duration_mu, duration_sigma)
# mu/sigma describe a log-normal distribution (right-skewed, like real traffic).
SERVICES = [
    (443, "TCP", 0.45, 3.4, 0.9, 700, 0.8, 0.9),    # HTTPS
    (80, "TCP", 0.15, 3.0, 0.8, 600, 0.6, 0.8),     # HTTP
    (53, "UDP", 0.18, 1.0, 0.3, 90, -3.0, 0.5),     # DNS
    (22, "TCP", 0.03, 4.0, 1.0, 150, 3.0, 1.0),     # SSH
    (587, "TCP", 0.04, 2.8, 0.7, 900, 0.5, 0.8),    # Email submission
    (8080, "TCP", 0.05, 3.2, 0.9, 650, 0.8, 0.9),   # Web proxy
    (3306, "TCP", 0.02, 4.0, 0.8, 400, 0.0, 0.8),   # Internal database
    (123, "UDP", 0.03, 0.7, 0.2, 76, -3.5, 0.4),    # Time sync (NTP)
    (0, "ICMP", 0.05, 1.3, 0.4, 84, 0.3, 0.3),      # Ping (ICMP has no ports)
]
SERVICE_PROBABILITIES = [s[2] for s in SERVICES]

# Busy during the day, quiet at night (24 hourly weights)
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 1, 2, 3, 5, 8, 9, 9, 9,
                         8, 8, 9, 9, 8, 7, 5, 4, 3, 2, 2, 1], dtype=float)

CAMPUS_SERVERS = [f"10.0.0.{i}" for i in range(10, 40)]
EXTERNAL_HOSTS = [f"203.0.113.{i}" for i in range(1, 120)] + \
                 [f"198.51.100.{i}" for i in range(1, 120)]
SCANNER_HOSTS = ["10.10.99.5", "10.10.99.6", "10.10.99.7"]   # synthetic "noisy" hosts
UNUSUAL_PORTS = [23, 445, 1337, 3389, 4444, 5900, 6667, 9001, 31337]


def _client_ip(rng):
    """A random synthetic student/staff device on the campus network."""
    return f"10.10.{rng.integers(0, 8)}.{rng.integers(2, 250)}"


def _normal_row(rng):
    """One normal-looking connection."""
    i = rng.choice(len(SERVICES), p=SERVICE_PROBABILITIES)
    port, proto, _, p_mu, p_sigma, bpp, d_mu, d_sigma = SERVICES[i]

    packets = max(1, int(round(rng.lognormal(p_mu, p_sigma))))
    avg_size = max(40.0, rng.normal(bpp, bpp * 0.2))
    duration = max(0.001, round(float(rng.lognormal(d_mu, d_sigma)), 3))

    # Local services are on campus servers, others are outside hosts
    if port in (3306, 53, 123):
        destination = str(rng.choice(CAMPUS_SERVERS))
    else:
        destination = str(rng.choice(EXTERNAL_HOSTS + CAMPUS_SERVERS))

    return {
        "source_ip": _client_ip(rng),
        "destination_ip": destination,
        "source_port": 0 if proto == "ICMP" else int(rng.integers(49152, 65536)),
        "destination_port": port,
        "protocol": proto,
        "packet_count": packets,
        "bytes_transferred": int(packets * avg_size),
        "duration": duration,
    }


def _anomaly_row(rng):
    """One synthetic anomalous connection (four different patterns)."""
    kind = rng.choice(4, p=[0.30, 0.25, 0.30, 0.15])

    if kind == 0:
        # Very high packet count (flood-like burst) towards a campus server
        packets = int(rng.integers(3000, 25000))
        return {
            "source_ip": str(rng.choice(EXTERNAL_HOSTS)),
            "destination_ip": str(rng.choice(CAMPUS_SERVERS[:3])),
            "source_port": int(rng.integers(1024, 65536)),
            "destination_port": int(rng.choice([80, 443])),
            "protocol": str(rng.choice(["TCP", "UDP"], p=[0.7, 0.3])),
            "packet_count": packets,
            "bytes_transferred": int(packets * rng.uniform(500, 1200)),
            "duration": round(float(rng.uniform(5, 60)), 3),
        }

    if kind == 1:
        # Unusually large data transfer (huge number of bytes, long duration)
        total_bytes = int(rng.integers(80_000_000, 600_000_000))
        return {
            "source_ip": _client_ip(rng),
            "destination_ip": str(rng.choice(EXTERNAL_HOSTS)),
            "source_port": int(rng.integers(49152, 65536)),
            "destination_port": int(rng.choice([443, 22])),
            "protocol": "TCP",
            "packet_count": int(total_bytes / rng.uniform(900, 1400)),
            "bytes_transferred": total_bytes,
            "duration": round(float(rng.uniform(60, 600)), 3),
        }

    if kind == 2:
        # Many very short connections to random ports from the same few hosts
        packets = int(rng.integers(1, 4))
        return {
            "source_ip": str(rng.choice(SCANNER_HOSTS)),
            "destination_ip": str(rng.choice(CAMPUS_SERVERS)),
            "source_port": int(rng.integers(1024, 65536)),
            "destination_port": int(rng.integers(1, 65536)),
            "protocol": "TCP",
            "packet_count": packets,
            "bytes_transferred": int(packets * rng.uniform(40, 60)),
            "duration": round(float(rng.uniform(0.001, 0.02)), 3),
        }

    # kind == 3: activity on an unusual destination port with moderate volume
    packets = max(1, int(round(rng.lognormal(3.5, 0.6))))
    return {
        "source_ip": _client_ip(rng),
        "destination_ip": str(rng.choice(EXTERNAL_HOSTS)),
        "source_port": int(rng.integers(49152, 65536)),
        "destination_port": int(rng.choice(UNUSUAL_PORTS)),
        "protocol": "TCP",
        "packet_count": packets,
        "bytes_transferred": int(packets * rng.uniform(300, 600)),
        "duration": round(float(rng.lognormal(1.0, 0.7)), 3),
    }


def _timestamps(rng, n, start, days, daytime_pattern):
    """Random timestamps over `days` days (busy daytime pattern or uniform)."""
    if daytime_pattern:
        weights = np.tile(HOUR_WEIGHTS, days)
        hours = rng.choice(24 * days, size=n, p=weights / weights.sum())
        seconds = hours * 3600 + rng.uniform(0, 3600, size=n)
    else:
        seconds = rng.uniform(0, days * 86400, size=n)
    return pd.Timestamp(start) + pd.to_timedelta(seconds, unit="s")


def generate_records(n_rows=8000, anomaly_rate=0.05, seed=42, start=None, days=3):
    """Create a DataFrame of synthetic traffic (normal + anomalous rows)."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 3, 2)

    n_anomalies = int(round(n_rows * anomaly_rate))
    n_normal = n_rows - n_anomalies

    normal = pd.DataFrame([_normal_row(rng) for _ in range(n_normal)])
    anomalous = pd.DataFrame([_anomaly_row(rng) for _ in range(n_anomalies)])

    normal["timestamp"] = _timestamps(rng, n_normal, start, days, True)
    anomalous["timestamp"] = _timestamps(rng, n_anomalies, start, days, False)
    normal["label"] = "normal"
    anomalous["label"] = "anomaly"

    df = pd.concat([normal, anomalous], ignore_index=True)
    df["timestamp"] = df["timestamp"].dt.floor("s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "source_ip", "destination_ip", "source_port",
               "destination_port", "protocol", "packet_count",
               "bytes_transferred", "duration", "label"]]


def generate_live_record(anomaly_probability=0.15, rng=None):
    """One new synthetic record stamped with the current time (Live Monitoring)."""
    rng = rng or np.random.default_rng()
    if rng.random() < anomaly_probability:
        row, label = _anomaly_row(rng), "anomaly"
    else:
        row, label = _normal_row(rng), "normal"
    row["timestamp"] = pd.Timestamp.now().floor("s")
    row["label"] = label
    return pd.DataFrame([row])[["timestamp", "source_ip", "destination_ip",
                                "source_port", "destination_port", "protocol",
                                "packet_count", "bytes_transferred",
                                "duration", "label"]]


def create_dataset_files(data_path=DATA_PATH, sample_path=SAMPLE_UPLOAD_PATH, n_rows=8000):
    """Write the training CSV (and optionally a small sample upload CSV)."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    generate_records(n_rows=n_rows, seed=42).to_csv(data_path, index=False)
    if sample_path is not None:
        sample = generate_records(n_rows=300, anomaly_rate=0.10, seed=7).drop(columns="label")
        sample.to_csv(sample_path, index=False)
    return data_path


if __name__ == "__main__":
    path = create_dataset_files()
    print(f"Synthetic dataset created: {path}")
    print(f"Sample upload file created: {SAMPLE_UPLOAD_PATH}")
