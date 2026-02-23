"""
train_common.py — Shared training utilities for IONIS models

This module contains all shared code for training:
- Model architecture: IonisGate (imported from model.py)
- MonotonicMLP for physics-constrained sidecars
- Feature engineering
- Grid utilities
- Dataset class
- Data loading from ClickHouse

Version-specific training scripts should:
1. Load their config from JSON
2. Import from this module
3. Call the training functions with their config
"""

import gc
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import clickhouse_connect

# Model architecture lives in model.py (no ClickHouse dependency)
from model import (  # noqa: F401
    MonotonicMLP, IonisGate, _gate,
    get_device, grid4_to_latlon, build_features,
    latlon_to_grid4, latlon_to_grid4_array,
    sfi_bucket, sfi_bucket_to_index,
    BAND_FREQ_HZ, GRID_RE,
)

log = logging.getLogger(__name__)


# ── Grid Utilities ───────────────────────────────────────────────────────────

def grid4_to_latlon_arrays(grids):
    """Convert array of 4-char grids to (lat, lon) arrays."""
    lats = np.zeros(len(grids), dtype=np.float32)
    lons = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        lats[i], lons[i] = grid4_to_latlon(g)
    return lats, lons


# ── IRI Atlas Loading ────────────────────────────────────────────────────────

def load_iri_atlas(npz_path):
    """Load IRI atlas from NumPy archive (CoW-safe).

    The atlas contains pre-computed IRI-2020 parameters (foF2, hmF2, foE)
    for every grid×hour×month×SFI combination.

    Args:
        npz_path: Path to iri_lookup.npz

    Returns:
        (data, grid_index, sfi_buckets) tuple:
        - data: (31692, 24, 12, 18, 3) float32 array
        - grid_index: dict mapping grid_4 string to row index
        - sfi_buckets: [70, 80, ..., 240] array

    Note:
        NumPy arrays share across fork() workers without copying.
        2 GB total, not 120 GB with Python dict.
    """
    npz = np.load(npz_path, allow_pickle=True)
    data = npz["data"]                      # (31692, 24, 12, 18, 3) float32
    grid_index = npz["grid_index"].item()   # dict: grid_4 -> int
    sfi_buckets = npz["sfi_buckets"]        # [70, 80, ..., 240]
    log.info(f"IRI atlas loaded: {data.shape}, {data.nbytes / 1e6:.0f} MB")
    return data, grid_index, sfi_buckets


# ── Feature Engineering ──────────────────────────────────────────────────────

def solar_elevation_vectorized(lat, lon, hour_utc, day_of_year):
    """
    Compute solar elevation angle in degrees (vectorized).

    Positive = sun above horizon (daylight)
    Negative = sun below horizon (night)

    Args:
        lat: array of latitudes in degrees
        lon: array of longitudes in degrees
        hour_utc: array of UTC hours
        day_of_year: array of day-of-year values (1-366)

    Returns:
        Array of solar elevation angles in degrees (-90 to +90)
    """
    # Solar declination
    dec = -23.44 * np.cos(np.radians(360.0 / 365.0 * (day_of_year + 10)))
    dec_r = np.radians(dec)
    lat_r = np.radians(lat)

    # Hour angle
    solar_hour = hour_utc + lon / 15.0
    ha_r = np.radians((solar_hour - 12.0) * 15.0)

    # Solar elevation
    sin_elev = (np.sin(lat_r) * np.sin(dec_r) +
                np.cos(lat_r) * np.cos(dec_r) * np.cos(ha_r))
    sin_elev = np.clip(sin_elev, -1.0, 1.0)

    return np.degrees(np.arcsin(sin_elev))


def _sigmoid(x):
    """Numpy sigmoid function for endpoint darkness."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_endpoint_darkness_vectorized(hour_utc, lon):
    """
    Compute darkness factor for endpoints (vectorized).

    Args:
        hour_utc: array of UTC hours
        lon: array of longitudes

    Returns:
        Array of darkness values [0, 1] where 1 = full darkness
    """
    local_hour = (hour_utc + lon / 15.0) % 24.0
    dist_from_noon = np.abs(local_hour - 12.0)
    return _sigmoid((dist_from_noon - 6.0) * 2.5)


def engineer_features(df, config):
    """
    Compute features from signature columns.

    Args:
        df: DataFrame with signature data
        config: Dict with model config (band_to_hz, input_dim, sfi_idx, kp_penalty_idx)

    Returns:
        np.ndarray of shape (n_rows, input_dim)
    """
    band_to_hz = config["band_to_hz"]
    input_dim = config["model"]["input_dim"]
    sfi_idx = config["model"]["sfi_idx"]
    kp_penalty_idx = config["model"]["kp_penalty_idx"]

    distance = df['avg_distance'].values.astype(np.float32)
    band = df['band'].values.astype(np.int32)
    hour = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)
    azimuth = df['avg_azimuth'].values.astype(np.float32)
    sfi = df['avg_sfi'].values.astype(np.float32)
    kp = df['avg_kp'].values.astype(np.float32)

    tx_lats, tx_lons = grid4_to_latlon_arrays(df['tx_grid_4'].values)
    rx_lats, rx_lons = grid4_to_latlon_arrays(df['rx_grid_4'].values)

    midpoint_lat = (tx_lats + rx_lats) / 2.0
    midpoint_lon = (tx_lons + rx_lons) / 2.0

    # Convert band_to_hz keys to int if they're strings (from JSON)
    if isinstance(list(band_to_hz.keys())[0], str):
        band_to_hz = {int(k): v for k, v in band_to_hz.items()}

    freq_hz = np.array([band_to_hz.get(b, 14_097_100) for b in band], dtype=np.float32)

    X = np.zeros((len(df), input_dim), dtype=np.float32)

    # Features 0-9: Core DNN inputs (geography/time) — always present
    X[:, 0] = distance / 20000.0
    X[:, 1] = np.log10(freq_hz) / 8.0
    X[:, 2] = np.sin(2.0 * np.pi * hour / 24.0)
    X[:, 3] = np.cos(2.0 * np.pi * hour / 24.0)
    X[:, 4] = np.sin(2.0 * np.pi * azimuth / 360.0)
    X[:, 5] = np.cos(2.0 * np.pi * azimuth / 360.0)
    X[:, 6] = np.abs(tx_lats - rx_lats) / 180.0
    X[:, 7] = midpoint_lat / 90.0
    X[:, 8] = np.sin(2.0 * np.pi * month / 12.0)
    X[:, 9] = np.cos(2.0 * np.pi * month / 12.0)

    dnn_dim = config["model"]["dnn_dim"]

    if dnn_dim >= 15:
        # V22: Solar elevation with band×darkness cross-products
        # Feature 10: vertex_lat
        # Feature 11-12: tx_solar_dep, rx_solar_dep
        # Feature 13-14: freq_x_tx_dark, freq_x_rx_dark

        # Get day_of_year from DataFrame (must be added to query)
        if 'day_of_year' in df.columns:
            day_of_year = df['day_of_year'].values.astype(np.float32)
        else:
            # Fallback: approximate from month (mid-month)
            log.warning("day_of_year not in DataFrame, using mid-month approximation")
            day_of_year = (month - 1) * 30.5 + 15

        # vertex_lat (index 10)
        azimuth_rad = np.radians(azimuth)
        tx_lat_rad = np.radians(tx_lats)
        vertex_lat_rad = np.arccos(np.abs(np.sin(azimuth_rad) * np.cos(tx_lat_rad)))
        X[:, 10] = np.degrees(vertex_lat_rad) / 90.0

        # Solar elevation at each endpoint (positive=day, negative=night)
        tx_solar = solar_elevation_vectorized(tx_lats, tx_lons, hour, day_of_year)
        rx_solar = solar_elevation_vectorized(rx_lats, rx_lons, hour, day_of_year)
        X[:, 11] = tx_solar / 90.0                          # tx_solar_dep normalized
        X[:, 12] = rx_solar / 90.0                          # rx_solar_dep normalized

        # Cross-products: band × darkness interaction
        # Centered around 10 MHz pivot — the ionospheric D/F-layer transition
        # Below 10 MHz: darkness helps (D-layer absorption vanishes)
        # Above 10 MHz: darkness kills (F-layer refraction vanishes)
        #
        # ASYMMETRIC SCALING (V22-gamma):
        # Linear pivot gave +0.90 for 10m but only -0.41 for 160m, causing
        # optimizer to ignore low bands (gradient signal 2x weaker).
        # Fix: scale both ends to exactly ±1.0 for equal gradient weight.
        freq_mhz = freq_hz / 1e6
        freq_centered = np.where(
            freq_mhz >= 10.0,
            (freq_mhz - 10.0) / 18.0,   # 10m (28 MHz) -> +1.0
            (freq_mhz - 10.0) / 8.2     # 160m (1.8 MHz) -> -1.0
        )
        X[:, 13] = freq_centered * X[:, 11]                 # freq_x_tx_dark
        X[:, 14] = freq_centered * X[:, 12]                 # freq_x_rx_dark

        # V23: IRI features (foF2, hmF2, foE)
        if dnn_dim >= 18:
            iri_data = config.get("_iri_data")
            iri_grid_index = config.get("_iri_grid_index")

            if iri_data is None or iri_grid_index is None:
                raise ValueError("V23 (dnn_dim>=18) requires IRI atlas. "
                                 "Load via load_iri_atlas() and set config['_iri_data'] and config['_iri_grid_index']")

            # Compute midpoint grids
            mid_lats = (tx_lats + rx_lats) / 2.0
            mid_lons = (tx_lons + rx_lons) / 2.0
            mid_grids = latlon_to_grid4_array(mid_lats, mid_lons)

            # Convert SFI to bucket index (0-17)
            sfi_raw = df['avg_sfi'].values.astype(np.float32)
            sfi_idxs = np.clip(np.round(sfi_raw / 10).astype(int) - 7, 0, 17)

            # Month to 0-indexed
            month_idxs = month.astype(int) - 1

            # Hour as int
            hour_int = hour.astype(int) % 24

            # Vectorized grid index lookup (default to index 0 for missing)
            grid_idxs = np.array([iri_grid_index.get(g, 0) for g in mid_grids])

            # Batch array indexing — no Python loop per row
            # iri_data shape: (N_grids, 24, 12, 18, 3) = (grid, hour, month, sfi_bucket, [foF2, hmF2, foE])
            iri_vals = iri_data[grid_idxs, hour_int, month_idxs, sfi_idxs, :]
            foF2_mid = iri_vals[:, 0]   # MHz
            hmF2_mid = iri_vals[:, 1]   # km
            foE_mid  = iri_vals[:, 2]   # MHz

            # foF2_freq_ratio is THE feature: when > 1, operating freq < MUF
            X[:, 15] = foF2_mid / freq_mhz                  # foF2_freq_ratio
            X[:, 16] = foE_mid / 10.0                       # foE_mid normalized
            X[:, 17] = hmF2_mid / 500.0                     # hmF2_mid normalized

    elif dnn_dim >= 13:
        # V21-beta: Physics gates replace day_night_est
        # Feature 10: mutual_darkness (tx_darkness × rx_darkness)
        # Feature 11: mutual_daylight ((1-tx_darkness) × (1-rx_darkness))
        # Feature 12: vertex_lat
        tx_darkness = compute_endpoint_darkness_vectorized(hour, tx_lons)
        rx_darkness = compute_endpoint_darkness_vectorized(hour, rx_lons)
        X[:, 10] = tx_darkness * rx_darkness                # mutual_darkness
        X[:, 11] = (1 - tx_darkness) * (1 - rx_darkness)    # mutual_daylight

        # vertex_lat = arccos(|sin(azimuth) * cos(tx_lat)|)
        azimuth_rad = np.radians(azimuth)
        tx_lat_rad = np.radians(tx_lats)
        vertex_lat_rad = np.arccos(np.abs(np.sin(azimuth_rad) * np.cos(tx_lat_rad)))
        X[:, 12] = np.degrees(vertex_lat_rad) / 90.0

    elif dnn_dim == 12:
        # V21-alpha: day_night_est + vertex_lat
        X[:, 10] = np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)

        # vertex_lat = arccos(|sin(azimuth) * cos(tx_lat)|)
        azimuth_rad = np.radians(azimuth)
        tx_lat_rad = np.radians(tx_lats)
        vertex_lat_rad = np.arccos(np.abs(np.sin(azimuth_rad) * np.cos(tx_lat_rad)))
        X[:, 11] = np.degrees(vertex_lat_rad) / 90.0

    else:
        # V20 and earlier: day_night_est only
        X[:, 10] = np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)

    # Sidecar inputs (solar physics) — indices from config
    X[:, sfi_idx] = sfi / 300.0
    X[:, kp_penalty_idx] = 1.0 - kp / 9.0  # kp_penalty

    return X


# ── Training Helpers (defibrillator, clamping, optimizer) ────────────────────

def init_defibrillator(model):
    """
    Apply defibrillator initialization to keep sidecars alive.

    CRITICAL: Call this after model creation, before training.

    This init:
        - Sets sidecar weights to uniform(0.8, 1.2) instead of random
        - Sets fc2.bias to -10.0 (strong initial offset)
        - Freezes fc1.bias (prevents collapse)
    """
    def wake_up_sidecar(layer):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, 0.8, 1.2)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    model.sun_sidecar.apply(wake_up_sidecar)
    model.storm_sidecar.apply(wake_up_sidecar)

    with torch.no_grad():
        model.sun_sidecar.fc2.bias.fill_(-10.0)
        model.storm_sidecar.fc2.bias.fill_(-10.0)

    # Freeze fc1 bias, keep fc2 bias learnable
    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True

    log.info("Defibrillator applied: sidecar weights uniform(0.8-1.2), fc2.bias=-10.0, fc1.bias frozen")


def clamp_sidecars(model):
    """
    Clamp sidecar weights to [0.5, 2.0] to prevent collapse.

    CRITICAL: Call this after every optimizer.step() during training.
    """
    with torch.no_grad():
        for sidecar in [model.sun_sidecar, model.storm_sidecar]:
            sidecar.fc1.weight.clamp_(0.5, 2.0)
            sidecar.fc2.weight.clamp_(0.5, 2.0)


def get_optimizer_groups(model, trunk_lr=1e-5, scaler_lr=5e-5, sidecar_lr=1e-3):
    """
    Get 6-group optimizer configuration.

    Args:
        model: IonisGate model
        trunk_lr: Learning rate for trunk and base_head
        scaler_lr: Learning rate for scaler heads (intermediate)
        sidecar_lr: Learning rate for sidecars (fastest)

    Returns:
        List of parameter groups for AdamW optimizer
    """
    return [
        {'params': model.trunk.parameters(), 'lr': trunk_lr},
        {'params': model.base_head.parameters(), 'lr': trunk_lr},
        {'params': model.sun_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': model.storm_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': [p for p in model.sun_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
    ]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SignatureDataset(Dataset):
    """PyTorch Dataset for signature data."""

    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_source_data(client, table, sample_size=None, where_clause="avg_sfi > 0",
                     include_day_of_year=False):
    """Load signature data from a ClickHouse table.

    Args:
        client: ClickHouse client
        table: Table name (e.g., 'wspr.signatures_v2_terrestrial')
        sample_size: Optional limit on rows to load
        where_clause: SQL WHERE clause
        include_day_of_year: If True, compute day_of_year from month (V22+)
            Note: Signature tables are aggregated and don't have timestamp.
            We use mid-month approximation: doy = (month - 1) * 30.5 + 15

    Returns:
        DataFrame with signature data
    """
    count = client.command(f"SELECT count() FROM {table}")
    log.info(f"{table}: {count:,} rows available")

    limit = f" LIMIT {sample_size}" if sample_size else ""
    order = " ORDER BY rand()" if sample_size else ""

    # V22+: Compute day_of_year from month (mid-month approximation)
    # Signature tables are aggregated by hour/month, no raw timestamp available
    doy_col = ", toUInt16((month - 1) * 30.5 + 15) AS day_of_year" if include_day_of_year else ""

    query = f"""
    SELECT
        tx_grid_4, rx_grid_4, band, hour, month{doy_col},
        median_snr, spot_count, snr_std, reliability,
        avg_sfi, avg_kp, avg_distance, avg_azimuth
    FROM {table}
    WHERE {where_clause}
    {order}{limit}
    """

    t0 = time.perf_counter()
    df = client.query_df(query)
    elapsed = time.perf_counter() - t0
    log.info(f"{table}: loaded {len(df):,} rows in {elapsed:.1f}s")

    return df


def load_combined_data(config):
    """
    Load all data sources and apply per-source normalization.

    Args:
        config: Dict with data configuration

    Returns:
        X_combined, y_combined, w_combined, date_range, norm_constants
    """
    ch_host = config["clickhouse"]["host"]
    ch_port = config["clickhouse"]["port"]

    wspr_sample = config["data"]["wspr_sample"]
    rbn_full_sample = config["data"]["rbn_full_sample"]
    rbn_dx_upsample = config["data"]["rbn_dx_upsample"]
    contest_upsample = config["data"]["contest_upsample"]
    storm_upsample = config["data"]["storm_upsample"]
    kp_penalty_idx = config["model"]["kp_penalty_idx"]

    log.info(f"Connecting to ClickHouse at {ch_host}:{ch_port}...")
    client = clickhouse_connect.get_client(host=ch_host, port=ch_port)

    # Load each source
    wspr_df = load_source_data(client, "wspr.signatures_v2_terrestrial", wspr_sample)

    # Guard: skip RBN Full when sample == 0 (V20 recipe)
    if rbn_full_sample > 0:
        rbn_full_df = load_source_data(client, "rbn.signatures", rbn_full_sample)
    else:
        log.info("rbn.signatures: skipped (rbn_full_sample=0)")
        rbn_full_df = None

    rbn_dx_df = load_source_data(client, "rbn.dxpedition_signatures", None)
    contest_df = load_source_data(client, "contest.signatures", None)

    # Get date range
    date_query = """
    SELECT
        formatDateTime(min(timestamp), '%Y-%m-%d') as min_date,
        formatDateTime(max(timestamp), '%Y-%m-%d') as max_date
    FROM wspr.bronze
    """
    date_result = client.query(date_query)
    min_date, max_date = date_result.result_rows[0]
    date_range = f"{min_date} to {max_date}"

    client.close()

    # ── Per-source normalization ──
    log.info("")
    log.info("=== PER-SOURCE NORMALIZATION (Rosetta Stone) ===")

    wspr_snr_raw = wspr_df['median_snr'].values.astype(np.float32)
    rbn_dx_snr_raw = rbn_dx_df['median_snr'].values.astype(np.float32)
    contest_snr_raw = contest_df['median_snr'].values.astype(np.float32)

    # Handle RBN Full (may be None if rbn_full_sample=0)
    if rbn_full_df is not None:
        rbn_full_snr_raw = rbn_full_df['median_snr'].values.astype(np.float32)
        rbn_all_snr = np.concatenate([rbn_full_snr_raw, rbn_dx_snr_raw])
    else:
        rbn_full_snr_raw = np.array([], dtype=np.float32)
        rbn_all_snr = rbn_dx_snr_raw  # DXpedition only

    norm_constants = {
        "wspr": {
            "mean": float(wspr_snr_raw.mean()),
            "std": float(wspr_snr_raw.std())
        },
        "rbn": {
            "mean": float(rbn_all_snr.mean()),
            "std": float(rbn_all_snr.std())
        },
        "contest": {
            "mean": float(contest_snr_raw.mean()),
            "std": float(contest_snr_raw.std())
        }
    }

    log.info("Per-source normalization constants:")
    log.info(f"  WSPR:    mean={norm_constants['wspr']['mean']:.2f} dB, std={norm_constants['wspr']['std']:.2f} dB")
    log.info(f"  RBN:     mean={norm_constants['rbn']['mean']:.2f} dB, std={norm_constants['rbn']['std']:.2f} dB")
    log.info(f"  Contest: mean={norm_constants['contest']['mean']:.2f} dB, std={norm_constants['contest']['std']:.2f} dB")

    # Normalize to Z-scores
    wspr_snr_z = (wspr_snr_raw - norm_constants['wspr']['mean']) / norm_constants['wspr']['std']
    rbn_dx_snr_z = (rbn_dx_snr_raw - norm_constants['rbn']['mean']) / norm_constants['rbn']['std']
    contest_snr_z = (contest_snr_raw - norm_constants['contest']['mean']) / norm_constants['contest']['std']

    # RBN Full Z-scores (empty array if skipped)
    if len(rbn_full_snr_raw) > 0:
        rbn_full_snr_z = (rbn_full_snr_raw - norm_constants['rbn']['mean']) / norm_constants['rbn']['std']
    else:
        rbn_full_snr_z = np.array([], dtype=np.float32)

    log.info("")
    log.info("Normalized Z-scores (should all be ~0 mean, ~1 std):")
    log.info(f"  WSPR:    mean={wspr_snr_z.mean():.4f}, std={wspr_snr_z.std():.4f}")
    if len(rbn_full_snr_z) > 0:
        log.info(f"  RBN:     mean={rbn_full_snr_z.mean():.4f}, std={rbn_full_snr_z.std():.4f}")
    else:
        log.info(f"  RBN:     (skipped - DXpedition only)")
    log.info(f"  Contest: mean={contest_snr_z.mean():.4f}, std={contest_snr_z.std():.4f}")

    # ── Engineer Features ──
    log.info("")
    log.info("Engineering features...")
    wspr_X = engineer_features(wspr_df, config)
    rbn_dx_X = engineer_features(rbn_dx_df, config)
    contest_X = engineer_features(contest_df, config)

    # RBN Full features (empty if skipped)
    if rbn_full_df is not None:
        rbn_full_X = engineer_features(rbn_full_df, config)
    else:
        input_dim = config["model"]["input_dim"]
        rbn_full_X = np.zeros((0, input_dim), dtype=np.float32)

    # ── Prepare weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w = wspr_w / wspr_w.mean()

    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w = rbn_dx_w / rbn_dx_w.mean()

    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w = contest_w / contest_w.mean()

    # RBN Full weights (empty if skipped)
    if rbn_full_df is not None:
        rbn_full_w = rbn_full_df['spot_count'].values.astype(np.float32)
        rbn_full_w = rbn_full_w / rbn_full_w.mean()
    else:
        rbn_full_w = np.array([], dtype=np.float32)

    # ── Upsample DXpedition ──
    log.info(f"Upsampling RBN DXpedition {rbn_dx_upsample}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (rbn_dx_upsample, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr_z, rbn_dx_upsample)
    rbn_dx_w_up = np.tile(rbn_dx_w, rbn_dx_upsample)
    log.info(f"  DXpedition effective rows: {len(rbn_dx_snr_up):,}")

    # ── Upsample Contest if needed ──
    if contest_upsample > 1:
        log.info(f"Upsampling Contest data {contest_upsample}x...")
        contest_X_up = np.tile(contest_X, (contest_upsample, 1))
        contest_snr_up = np.tile(contest_snr_z, contest_upsample)
        contest_w_up = np.tile(contest_w, contest_upsample)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr_z
        contest_w_up = contest_w
    log.info(f"  Contest effective rows: {len(contest_snr_up):,}")

    # ── Combine all sources ──
    X_combined = np.vstack([wspr_X, rbn_full_X, rbn_dx_X_up, contest_X_up])
    y_combined = np.concatenate([wspr_snr_z, rbn_full_snr_z, rbn_dx_snr_up, contest_snr_up])
    w_combined = np.concatenate([wspr_w, rbn_full_w, rbn_dx_w_up, contest_w_up])

    total_before_storm = len(X_combined)
    log.info("")
    log.info(f"Combined dataset (before storm upsample): {total_before_storm:,} rows")
    log.info(f"  WSPR (floor):       {len(wspr_X):,} ({100*len(wspr_X)/total_before_storm:.1f}%)")
    log.info(f"  RBN Full (middle):  {len(rbn_full_X):,} ({100*len(rbn_full_X)/total_before_storm:.1f}%)")
    log.info(f"  RBN DX (rare):      {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/total_before_storm:.1f}%)")
    log.info(f"  Contest (ceiling):  {len(contest_snr_up):,} ({100*len(contest_snr_up)/total_before_storm:.1f}%)")

    # ── Storm upsample ──
    storm_threshold = 4.0 / 9.0  # kp >= 5
    storm_mask = X_combined[:, kp_penalty_idx] <= storm_threshold
    storm_count = storm_mask.sum()

    log.info(f"Storm upsample: {storm_count:,} rows with Kp >= 5 ({100*storm_count/total_before_storm:.2f}%)")

    if storm_count > 0 and storm_upsample > 1:
        storm_X = X_combined[storm_mask]
        storm_y = y_combined[storm_mask]
        storm_w = w_combined[storm_mask]

        log.info(f"  Upsampling storm rows {storm_upsample}x...")
        X_combined = np.vstack([X_combined] + [storm_X] * storm_upsample)
        y_combined = np.concatenate([y_combined] + [storm_y] * storm_upsample)
        w_combined = np.concatenate([w_combined] + [storm_w] * storm_upsample)

        log.info(f"  After storm upsample: {len(X_combined):,} rows (+{storm_count * storm_upsample:,})")

    log.info("")
    log.info(f"Combined targets (Z-scores): mean={y_combined.mean():.4f}, std={y_combined.std():.4f}")

    y_combined = y_combined.reshape(-1, 1)
    w_combined = w_combined.reshape(-1, 1)

    # Free DataFrames
    del wspr_df, rbn_dx_df, contest_df
    if rbn_full_df is not None:
        del rbn_full_df
    del wspr_X, rbn_full_X, rbn_dx_X, contest_X, rbn_dx_X_up, contest_X_up
    del wspr_snr_raw, rbn_full_snr_raw, rbn_dx_snr_raw, contest_snr_raw
    gc.collect()

    return X_combined, y_combined, w_combined, date_range, norm_constants


# ── Config Logging ───────────────────────────────────────────────────────────

def log_config(config, config_file, device):
    """Log all configuration at the start of training."""

    log.info(f"{'='*70}")
    log.info(f"IONIS {config['version']} | {config['phase']}")
    log.info(f"{'='*70}")
    log.info(f"Config: {config_file}")
    log.info("")

    log.info("=== TRAINING ===")
    log.info(f"  Device: {device}")
    log.info(f"  Epochs: {config['training']['epochs']}")
    log.info(f"  Batch size: {config['training']['batch_size']:,}")
    log.info(f"  Validation split: {config['training']['val_split']:.0%}")
    log.info("")

    log.info("=== DATA ===")
    log.info(f"  WSPR sample: {config['data']['wspr_sample']:,}")
    log.info(f"  RBN Full sample: {config['data']['rbn_full_sample']:,}")
    log.info(f"  RBN DX upsample: {config['data']['rbn_dx_upsample']}x")
    log.info(f"  Contest upsample: {config['data']['contest_upsample']}x")
    log.info(f"  Storm upsample: {config['data']['storm_upsample']}x (Kp >= 5)")
    log.info("")

    log.info("=== LEARNING RATES ===")
    trunk_lr = config['training']['trunk_lr']
    sidecar_lr = config['training']['sidecar_lr']
    log.info(f"  Trunk + Gates: {trunk_lr}")
    log.info(f"  Sidecars: {sidecar_lr} ({sidecar_lr/trunk_lr:.0f}x boost)")
    log.info(f"  Weight decay: {config['training']['weight_decay']}")
    log.info("")

    log.info("=== MODEL ===")
    log.info(f"  Architecture: {config['model']['architecture']}")
    log.info(f"  DNN dim: {config['model']['dnn_dim']}")
    hidden = config['model']['hidden_dim']
    log.info(f"  Hidden: {hidden} (trunk: {hidden*2}→{hidden}→{hidden//2}→1)")
    log.info(f"  Sidecar hidden: {config['model']['sidecar_hidden']}")
    log.info("")
