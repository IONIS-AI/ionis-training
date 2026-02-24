#!/usr/bin/env python3
"""
train_audit03.py — AUDIT-03: 35-Bucket IRI Atlas (5-Unit SFI Steps)

HYPOTHESIS: Finer SFI resolution in IRI atlas may improve foF2 feature accuracy
            and affect sidecar behavior.

RECIPE: V23 (dnn_dim=18, IRI features: foF2_freq_ratio, foE_mid, hmF2_mid)
CHANGE: Use 35-bucket IRI atlas (5-unit SFI steps: 70, 75, 80...240)
        instead of 18-bucket (10-unit steps: 70, 80, 90...240)
CLAMP: Original [0.5, 2.0]
DURATION: 100 epochs

The 35-bucket atlas has 2x finer SFI resolution:
  - 18-bucket: sfi_idx = (sfi / 10) - 7, range 0-17
  - 35-bucket: sfi_idx = (sfi - 70) / 5, range 0-34
"""

import gc
import json
import logging
import os
import socket
import sys
import time
import uuid

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from safetensors.torch import save_file as save_safetensors

import clickhouse_connect

# Add parent for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
TRAINING_DIR = os.path.dirname(VERSIONS_DIR)  # ionis-training root
sys.path.insert(0, VERSIONS_DIR)
sys.path.insert(0, COMMON_DIR)

from common.train_common import (
    IonisGate,
    SignatureDataset,
    init_defibrillator,
    clamp_sidecars,
    get_optimizer_groups,
    load_source_data,
    load_iri_atlas,
    log_config,
    grid4_to_latlon,
    latlon_to_grid4_array,
    solar_elevation_vectorized,
)

# ── AUDIT-03 SPECIFIC: 35-Bucket Feature Engineering ─────────────────────────


def engineer_features_35bucket(df, config):
    """
    Compute features from signature columns with 35-bucket IRI indexing.

    AUDIT-03 CHANGE: SFI bucket indexing uses 5-unit steps instead of 10-unit.
    Formula: (sfi - 70) // 5, range 0-34 (35 buckets)

    Args:
        df: DataFrame with signature data
        config: Dict with model config

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

    # Grid to lat/lon
    tx_lats = np.zeros(len(df), dtype=np.float32)
    tx_lons = np.zeros(len(df), dtype=np.float32)
    rx_lats = np.zeros(len(df), dtype=np.float32)
    rx_lons = np.zeros(len(df), dtype=np.float32)
    for i, (tg, rg) in enumerate(zip(df['tx_grid_4'].values, df['rx_grid_4'].values)):
        tx_lats[i], tx_lons[i] = grid4_to_latlon(tg)
        rx_lats[i], rx_lons[i] = grid4_to_latlon(rg)

    midpoint_lat = (tx_lats + rx_lats) / 2.0
    midpoint_lon = (tx_lons + rx_lons) / 2.0

    # Convert band_to_hz keys to int if they're strings (from JSON)
    if isinstance(list(band_to_hz.keys())[0], str):
        band_to_hz = {int(k): v for k, v in band_to_hz.items()}

    freq_hz = np.array([band_to_hz.get(b, 14_097_100) for b in band], dtype=np.float32)

    X = np.zeros((len(df), input_dim), dtype=np.float32)

    # Features 0-9: Core DNN inputs (geography/time)
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

    # Get day_of_year from DataFrame
    if 'day_of_year' in df.columns:
        day_of_year = df['day_of_year'].values.astype(np.float32)
    else:
        day_of_year = (month - 1) * 30.5 + 15

    # V22+ Features 10-14: vertex_lat, solar elevation, band×darkness
    azimuth_rad = np.radians(azimuth)
    tx_lat_rad = np.radians(tx_lats)
    vertex_lat_rad = np.arccos(np.abs(np.sin(azimuth_rad) * np.cos(tx_lat_rad)))
    X[:, 10] = np.degrees(vertex_lat_rad) / 90.0  # vertex_lat

    tx_solar = solar_elevation_vectorized(tx_lats, tx_lons, hour, day_of_year)
    rx_solar = solar_elevation_vectorized(rx_lats, rx_lons, hour, day_of_year)
    X[:, 11] = tx_solar / 90.0  # tx_solar_dep
    X[:, 12] = rx_solar / 90.0  # rx_solar_dep

    # Asymmetric scaling for band×darkness cross-products
    freq_mhz = freq_hz / 1e6
    freq_centered = np.where(
        freq_mhz >= 10.0,
        (freq_mhz - 10.0) / 18.0,   # 10m (28 MHz) -> +1.0
        (freq_mhz - 10.0) / 8.2     # 160m (1.8 MHz) -> -1.0
    )
    X[:, 13] = freq_centered * X[:, 11]  # freq_x_tx_dark
    X[:, 14] = freq_centered * X[:, 12]  # freq_x_rx_dark

    # V23 Features 15-17: IRI atlas lookup with 35-BUCKET INDEXING
    iri_data = config.get("_iri_data")
    iri_grid_index = config.get("_iri_grid_index")

    if iri_data is None or iri_grid_index is None:
        raise ValueError("AUDIT-03 requires IRI atlas. "
                         "Load via load_iri_atlas() and set config['_iri_data'] and config['_iri_grid_index']")

    # Compute midpoint grids
    mid_lats = (tx_lats + rx_lats) / 2.0
    mid_lons = (tx_lons + rx_lons) / 2.0
    mid_grids = latlon_to_grid4_array(mid_lats, mid_lons)

    # ═══════════════════════════════════════════════════════════════════════
    # AUDIT-03 CHANGE: 35-bucket SFI indexing (5-unit steps)
    # ═══════════════════════════════════════════════════════════════════════
    # 18-bucket (V23):  sfi_idx = (sfi / 10) - 7, buckets 70, 80, 90...240
    # 35-bucket (A-03): sfi_idx = (sfi - 70) / 5, buckets 70, 75, 80...240
    sfi_raw = df['avg_sfi'].values.astype(np.float32)
    sfi_idxs = np.clip(((sfi_raw - 70) / 5).astype(int), 0, 34)
    # ═══════════════════════════════════════════════════════════════════════

    # Month to 0-indexed
    month_idxs = month.astype(int) - 1

    # Hour as int
    hour_int = hour.astype(int) % 24

    # Vectorized grid index lookup
    grid_idxs = np.array([iri_grid_index.get(g, 0) for g in mid_grids])

    # Batch array indexing
    # iri_data shape: (N_grids, 24, 12, 35, 3) = (grid, hour, month, sfi_bucket, [foF2, hmF2, foE])
    iri_vals = iri_data[grid_idxs, hour_int, month_idxs, sfi_idxs, :]
    foF2_mid = iri_vals[:, 0]   # MHz
    hmF2_mid = iri_vals[:, 1]   # km
    foE_mid  = iri_vals[:, 2]   # MHz

    # foF2_freq_ratio is THE feature: when > 1, operating freq < MUF
    X[:, 15] = foF2_mid / freq_mhz                  # foF2_freq_ratio
    X[:, 16] = foE_mid / 10.0                       # foE_mid normalized
    X[:, 17] = hmF2_mid / 500.0                     # hmF2_mid normalized

    # Sidecar inputs (solar physics) — indices from config
    X[:, sfi_idx] = sfi / 300.0
    X[:, kp_penalty_idx] = 1.0 - kp / 9.0  # kp_penalty

    return X


def log_sidecar_weights(model, epoch, log):
    """Log sun sidecar weight statistics for auditing."""
    with torch.no_grad():
        fc1_w = model.sun_sidecar.fc1.weight.cpu().numpy()
        fc2_w = model.sun_sidecar.fc2.weight.cpu().numpy()
        log.info(f"  SUN SIDECAR WEIGHTS (epoch {epoch}):")
        log.info(f"    fc1: min={fc1_w.min():.4f}, max={fc1_w.max():.4f}, mean={fc1_w.mean():.4f}")
        log.info(f"    fc2: min={fc2_w.min():.4f}, max={fc2_w.max():.4f}, mean={fc2_w.mean():.4f}")

        storm_fc1 = model.storm_sidecar.fc1.weight.cpu().numpy()
        storm_fc2 = model.storm_sidecar.fc2.weight.cpu().numpy()
        log.info(f"  STORM SIDECAR WEIGHTS (epoch {epoch}):")
        log.info(f"    fc1: min={storm_fc1.min():.4f}, max={storm_fc1.max():.4f}")
        log.info(f"    fc2: min={storm_fc2.min():.4f}, max={storm_fc2.max():.4f}")


# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_audit03.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Training Audit Trail ─────────────────────────────────────────────────────

AUDIT_HOST = "http://10.60.1.1:8123"


def audit_start(run_id: str, config: dict):
    """Record training run start in ClickHouse."""
    try:
        # Filter out underscore-prefixed keys (like _iri_data ndarray) before JSON
        config_safe = {k: v for k, v in config.items() if not k.startswith('_')}
        config_json = json.dumps(config_safe).replace('"', '\\"')
        features_array = "['" + "','".join(config["features"]) + "']"
        query = f"""
            INSERT INTO training.runs (
                run_id, version, variant, status, hostname,
                started_at, config_json,
                dnn_dim, hidden_dim, input_dim, training_epochs, batch_size,
                trunk_lr, sidecar_lr,
                wspr_sample, rbn_full_sample, rbn_dx_upsample, contest_upsample,
                features, notes
            ) VALUES (
                '{run_id}', '{config["version"]}', '{config.get("variant", "")}', 'running', '{socket.gethostname()}',
                now(),
                '{config_json}',
                {config["model"]["dnn_dim"]}, {config["model"]["hidden_dim"]},
                {config["model"]["input_dim"]}, {config["training"]["epochs"]},
                {config["training"]["batch_size"]},
                {config["training"]["trunk_lr"]}, {config["training"]["sidecar_lr"]},
                {config["data"]["wspr_sample"]}, {config["data"].get("rbn_full_sample", 0)},
                {config["data"]["rbn_dx_upsample"]}, {config["data"]["contest_upsample"]},
                {features_array},
                'AUDIT-03: 35-bucket IRI atlas (5-unit SFI steps)'
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit start failed: {e}")


def audit_epoch(run_id: str, epoch: int, train_loss: float, val_loss: float,
                val_rmse: float, val_pearson: float, sfi_benefit: float,
                storm_cost: float, epoch_seconds: float, is_best: bool):
    """Record epoch metrics in ClickHouse."""
    try:
        query = f"""
            INSERT INTO training.epochs (
                run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                sfi_benefit, storm_cost, epoch_seconds, is_best
            ) VALUES (
                '{run_id}', {epoch}, {train_loss}, {val_loss}, {val_rmse},
                {val_pearson}, {sfi_benefit}, {storm_cost}, {epoch_seconds}, {1 if is_best else 0}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception:
        pass


def audit_complete(run_id: str, config: dict, train_rows: int, val_rows: int,
                   date_range: str, val_rmse: float, val_pearson: float,
                   sfi_benefit: float, storm_cost: float, best_epoch: int,
                   wall_seconds: float):
    """Record training completion in ClickHouse."""
    try:
        # Filter out underscore-prefixed keys (like _iri_data ndarray) before JSON
        config_safe = {k: v for k, v in config.items() if not k.startswith('_')}
        config_json = json.dumps(config_safe).replace('"', '\\"')
        features_array = "['" + "','".join(config["features"]) + "']"
        date_min, date_max = date_range.split(" to ")
        query = f"""
            INSERT INTO training.runs (
                run_id, version, variant, status, hostname,
                started_at, completed_at, wall_seconds,
                config_json,
                dnn_dim, hidden_dim, input_dim, training_epochs, batch_size,
                trunk_lr, sidecar_lr,
                wspr_sample, rbn_full_sample, rbn_dx_upsample, contest_upsample,
                total_train_rows, total_val_rows, data_date_min, data_date_max,
                features,
                val_rmse, val_pearson, sfi_benefit, storm_cost, best_epoch,
                notes
            ) VALUES (
                '{run_id}', '{config["version"]}', '{config.get("variant", "")}', 'completed', '{socket.gethostname()}',
                now() - INTERVAL {int(wall_seconds)} SECOND, now(), {wall_seconds},
                '{config_json}',
                {config["model"]["dnn_dim"]}, {config["model"]["hidden_dim"]},
                {config["model"]["input_dim"]}, {config["training"]["epochs"]},
                {config["training"]["batch_size"]},
                {config["training"]["trunk_lr"]}, {config["training"]["sidecar_lr"]},
                {config["data"]["wspr_sample"]}, {config["data"].get("rbn_full_sample", 0)},
                {config["data"]["rbn_dx_upsample"]}, {config["data"]["contest_upsample"]},
                {train_rows}, {val_rows}, '{date_min}', '{date_max}',
                {features_array},
                {val_rmse}, {val_pearson}, {sfi_benefit}, {storm_cost}, {best_epoch},
                'AUDIT-03: 35-bucket atlas - COMPLETED'
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit complete failed: {e}")


# ── Per-Band Normalization ─────────────────────────────────────────────────

def normalize_snr_per_band(df, source, norm_constants):
    """Apply per-source per-band Z-score normalization."""
    snr = df['median_snr'].values.astype(np.float32).copy()
    band = df['band'].values

    for band_str, sources in norm_constants.items():
        b = int(band_str)
        mask = band == b
        if mask.sum() > 0 and source in sources:
            mean = sources[source]['mean']
            std = sources[source]['std']
            snr[mask] = (snr[mask] - mean) / std

    return snr


def pearson_r(pred, target):
    """Pearson correlation coefficient."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training function."""
    run_id = str(uuid.uuid4())
    training_start_time = time.perf_counter()

    log.info("=" * 70)
    log.info("AUDIT-03: 35-BUCKET IRI ATLAS (5-UNIT SFI STEPS)")
    log.info("=" * 70)
    log.info("")
    log.info("HYPOTHESIS: Finer SFI resolution may improve foF2 feature accuracy.")
    log.info("RECIPE: V23 + 35-bucket atlas (5-unit SFI steps: 70, 75, 80...240)")
    log.info("CLAMP: Original [0.5, 2.0]")
    log.info("")

    log_config(CONFIG, CONFIG_FILE, DEVICE)

    # ── Load IRI Atlas (35-bucket) ──
    log.info("")
    log.info("=== IRI ATLAS (AUDIT-03: 35-BUCKET) ===")
    iri_path = os.path.join(TRAINING_DIR, CONFIG["iri_atlas"]["path"])
    if not os.path.exists(iri_path):
        log.error(f"IRI atlas not found: {iri_path}")
        log.error("Run: scp 10.60.1.1:/mnt/ai-stack/ionis-ai/iri_lookup_35.npz data/")
        return

    iri_data, iri_grid_index, iri_sfi_buckets = load_iri_atlas(iri_path)
    log.info(f"  Grids: {len(iri_grid_index):,}")
    log.info(f"  SFI buckets: {len(iri_sfi_buckets)} (5-unit steps)")
    log.info(f"  SFI range: {iri_sfi_buckets[0]} to {iri_sfi_buckets[-1]}")
    log.info(f"  Shape: {iri_data.shape} (grid, hour, month, sfi, [foF2, hmF2, foE])")

    # Verify 35 buckets
    if len(iri_sfi_buckets) != 35:
        log.error(f"Expected 35 SFI buckets, got {len(iri_sfi_buckets)}")
        log.error("This is the 18-bucket atlas. Get the 35-bucket version from 9975WX.")
        return

    # Make atlas available to engineer_features via CONFIG
    CONFIG["_iri_data"] = iri_data
    CONFIG["_iri_grid_index"] = iri_grid_index

    audit_start(run_id, CONFIG)

    # ── Load Data ──
    ch = CONFIG["clickhouse"]
    client = clickhouse_connect.get_client(host=ch["host"], port=ch["port"])

    wspr_df = load_source_data(client, "wspr.signatures_v2_terrestrial",
                               CONFIG["data"]["wspr_sample"],
                               include_day_of_year=True)
    rbn_dx_df = load_source_data(client, "rbn.dxpedition_signatures", None,
                                  include_day_of_year=True)
    contest_df = load_source_data(client, "contest.signatures", None,
                                   include_day_of_year=True)

    date_result = client.query("""
        SELECT formatDateTime(min(timestamp), '%Y-%m-%d'),
               formatDateTime(max(timestamp), '%Y-%m-%d')
        FROM wspr.bronze
    """)
    date_range = f"{date_result.result_rows[0][0]} to {date_result.result_rows[0][1]}"
    client.close()

    # ── Per-Band Normalization ──
    norm_consts = CONFIG["norm_constants_per_band"]
    contest_src = CONFIG["data"]["contest_norm_source"]

    log.info("")
    log.info("=== PER-SOURCE PER-BAND NORMALIZATION ===")

    wspr_snr = normalize_snr_per_band(wspr_df, 'wspr', norm_consts)
    rbn_dx_snr = normalize_snr_per_band(rbn_dx_df, 'rbn', norm_consts)
    contest_snr = normalize_snr_per_band(contest_df, contest_src, norm_consts)

    log.info(f"  WSPR:    mean={wspr_snr.mean():.3f}, std={wspr_snr.std():.3f}")
    log.info(f"  RBN DX:  mean={rbn_dx_snr.mean():.3f}, std={rbn_dx_snr.std():.3f}")
    log.info(f"  Contest: mean={contest_snr.mean():.3f}, std={contest_snr.std():.3f}")

    # ── Features (35-bucket IRI) ──
    log.info("")
    log.info("Engineering features (35-BUCKET IRI INDEXING)...")
    log.info(f"  dnn_dim={CONFIG['model']['dnn_dim']} (V23 + 35-bucket atlas)")

    wspr_X = engineer_features_35bucket(wspr_df, CONFIG)
    rbn_dx_X = engineer_features_35bucket(rbn_dx_df, CONFIG)
    contest_X = engineer_features_35bucket(contest_df, CONFIG)

    # Spot-check IRI features
    log.info(f"  foF2_freq_ratio sample: min={wspr_X[:, 15].min():.3f}, "
             f"max={wspr_X[:, 15].max():.3f}, mean={wspr_X[:, 15].mean():.3f}")

    # NaN guard
    if np.isnan(wspr_X).any():
        log.error("NaN detected in WSPR features! Check IRI atlas for missing grids.")
        return

    # ── Weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w /= wspr_w.mean()
    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w /= rbn_dx_w.mean()
    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w /= contest_w.mean()

    # ── Upsample ──
    rbn_dx_up = CONFIG["data"]["rbn_dx_upsample"]
    contest_up = CONFIG["data"]["contest_upsample"]

    log.info(f"Upsampling RBN DXpedition {rbn_dx_up}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (rbn_dx_up, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr, rbn_dx_up)
    rbn_dx_w_up = np.tile(rbn_dx_w, rbn_dx_up)

    if contest_up > 1:
        log.info(f"Upsampling Contest {contest_up}x...")
        contest_X_up = np.tile(contest_X, (contest_up, 1))
        contest_snr_up = np.tile(contest_snr, contest_up)
        contest_w_up = np.tile(contest_w, contest_up)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr
        contest_w_up = contest_w

    # ── Combine ──
    X = np.vstack([wspr_X, rbn_dx_X_up, contest_X_up])
    y = np.concatenate([wspr_snr, rbn_dx_snr_up, contest_snr_up]).reshape(-1, 1)
    w = np.concatenate([wspr_w, rbn_dx_w_up, contest_w_up]).reshape(-1, 1)

    n = len(X)
    log.info("")
    log.info(f"Combined dataset: {n:,} rows")
    log.info(f"  WSPR:     {len(wspr_X):,} ({100*len(wspr_X)/n:.1f}%)")
    log.info(f"  RBN DX:   {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/n:.1f}%)")
    log.info(f"  Contest:  {len(contest_snr_up):,} ({100*len(contest_snr_up)/n:.1f}%)")
    log.info(f"Normalized SNR: mean={y.mean():.3f}, std={y.std():.3f}")
    log.info(f"Source data range: {date_range}")

    del wspr_df, rbn_dx_df, contest_df
    del wspr_X, rbn_dx_X, contest_X, rbn_dx_X_up, contest_X_up
    gc.collect()

    # ── Dataset + Split ──
    dataset = SignatureDataset(X, y, w)
    val_size = int(n * CONFIG["training"]["val_split"])
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model ──
    model = IonisGate(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        sfi_idx=CONFIG["model"]["sfi_idx"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisGate ({total_params:,} params)")

    # ── Defibrillator (standard init) ──
    init_defibrillator(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

    # Log initial weights
    log.info("")
    log.info("=== INITIAL WEIGHTS (POST-DEFIBRILLATOR) ===")
    log_sidecar_weights(model, 0, log)

    # ── Optimizer ──
    param_groups = get_optimizer_groups(
        model,
        trunk_lr=CONFIG["training"]["trunk_lr"],
        scaler_lr=CONFIG["training"]["scaler_lr"],
        sidecar_lr=CONFIG["training"]["sidecar_lr"],
    )
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG["training"]["weight_decay"])

    epochs = CONFIG["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = nn.HuberLoss(reduction='none', delta=CONFIG["training"]["huber_delta"])
    lambda_var = CONFIG["training"]["lambda_var"]

    checkpoint_name = CONFIG["checkpoint"]
    model_path = os.path.join(SCRIPT_DIR, checkpoint_name)
    meta_path = model_path.replace(".safetensors", "_meta.json")

    best_val_loss = float('inf')
    best_pearson = -1.0
    best_kp = 0.0
    best_sfi = 0.0
    best_epoch = 0

    # ── Training Loop ──
    log.info("")
    log.info(f"Training started ({epochs} epochs)")
    log.info(f"*** USING ORIGINAL CLAMP [0.5, 2.0] ***")
    log.info("")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>6s}  {'Kp9-':>5s}  {'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()

            out, sun_gate, storm_gate = model.forward_with_gates(bx)
            primary_loss = (criterion(out, by) * bw).mean()
            var_loss = -lambda_var * (sun_gate.var() + storm_gate.var())
            loss = primary_loss + var_loss

            loss.backward()
            optimizer.step()

            # Original clamp [0.5, 2.0]
            clamp_sidecars(model)

            train_loss_sum += primary_loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bx, by, bw in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss_sum += criterion(out, by).mean().item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)
        val_pearson = pearson_r(torch.cat(all_preds), torch.cat(all_targets))

        sfi_benefit = (model.get_sun_effect(200.0 / 300.0, DEVICE) -
                       model.get_sun_effect(70.0 / 300.0, DEVICE))
        storm_cost = (model.get_storm_effect(1.0, DEVICE) -
                      model.get_storm_effect(0.0, DEVICE))

        scheduler.step()
        epoch_sec = time.perf_counter() - t0

        is_best = val_loss < best_val_loss
        marker = ""
        if is_best:
            best_val_loss = val_loss
            best_pearson = val_pearson
            best_kp = storm_cost
            best_sfi = sfi_benefit
            best_epoch = epoch

            save_safetensors(model.state_dict(), model_path)

            metadata = {
                'version': CONFIG['version'],
                'variant': CONFIG['variant'],
                'audit': 'AUDIT-03: 35-bucket IRI atlas (5-unit SFI steps)',
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_benefit': sfi_benefit,
                'storm_cost': storm_cost,
                'iri_atlas': CONFIG["iri_atlas"]["path"],
                'sfi_buckets': 35,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+5.3f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        # Log weights every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            log_sidecar_weights(model, epoch, log)

        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    sfi_benefit, storm_cost, epoch_sec, is_best)

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    wall_seconds = time.perf_counter() - training_start_time

    audit_complete(run_id, CONFIG, train_size, val_size, date_range,
                   best_rmse, best_pearson, best_sfi, best_kp, best_epoch, wall_seconds)

    log.info("-" * len(hdr))
    log.info("")
    log.info("=" * 70)
    log.info("AUDIT-03 RESULTS")
    log.info("=" * 70)

    log.info("")
    log.info("=== FINAL WEIGHTS ===")
    log_sidecar_weights(model, epochs, log)

    log.info("")
    log.info("AUDIT-03 SUMMARY:")
    log.info(f"  Best RMSE: {best_rmse:.4f}σ")
    log.info(f"  Best Pearson: {best_pearson:+.4f}")
    log.info(f"  Final SFI+: {best_sfi:+.4f}σ")
    log.info(f"  Final Kp9-: {best_kp:+.4f}σ")
    log.info(f"  35-bucket atlas: {CONFIG['iri_atlas']['path']}")

    log.info("")
    log.info("COMPARISON TO V23 (18-bucket):")
    log.info("  Watch for differences in foF2_freq_ratio feature range/distribution.")
    log.info("  Finer SFI resolution may provide more precise ionospheric parameters.")

    log.info("")
    log.info(f"Checkpoint: {model_path}")
    log.info(f"Wall time: {wall_seconds/60:.1f} minutes")


if __name__ == '__main__':
    train()
