#!/usr/bin/env python3
"""
train_v2_core.py — IONIS V2 Oracle: Streaming ClickHouse Training Engine

Phase 5: Solar Fidelity — 15-feature model with SFI + Kp
Streams wspr.spots_raw + solar.indices_raw (SSN, SFI, 3-hourly Kp) from
ClickHouse (9975WX) directly to PyTorch tensors. No intermediate files.

Features 14-15 (new):
    prop_quality     = SFI_norm × (1 - Kp_norm)   — ionospheric quality
    band_sfi_interact = SFI_norm × freq_log        — band-dependent SFI effect

Usage:
    python train_v2_core.py
    CH_HOST=10.0.0.1 python train_v2_core.py

Environment:
    CH_HOST  ClickHouse host (default: 10.60.1.1)
    CH_PORT  ClickHouse HTTP port (default: 8123)
"""

import os
import sys
import time
import logging
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import clickhouse_connect


# ═══════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════

CH_HOST = os.environ.get('CH_HOST', '10.60.1.1')
CH_PORT = int(os.environ.get('CH_PORT', '8123'))
CH_CONNECT_TIMEOUT = 30
CH_QUERY_TIMEOUT = 600       # 10 min for large per-band queries
MAX_RETRIES = 3

# ClickHouse server-side memory settings for training sessions
# Note: 80G = ~74.5 GiB which the server accepts. Dense bands (40m, 20m)
# require ORDER BY over hundreds of millions of joined rows before LIMIT
# kicks in — external sort spills that to NVMe instead of RAM.
CH_SETTINGS = {
    'max_block_size': 524_288,                       # 512k rows per streamed block
    'max_bytes_before_external_group_by': 20_000_000_000,  # 20G — spill GROUP BY to NVMe
    'max_bytes_before_external_sort': 10_000_000_000,      # 10G — spill ORDER BY to NVMe
    'max_memory_usage': 80_000_000_000,              # 80G (~74.5 GiB)
    'max_threads': 16,                               # limit scan parallelism per query
}

BATCH_SIZE = 4096            # PyTorch training mini-batch
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # AdamW L2 regularization
SAMPLE_SIZE = 10_000_000     # Pilot: 10M rows
VAL_SPLIT = 0.2

INPUT_DIM = 15
HIDDEN_DIM = 256
DATE_START = '2020-01-01'
DATE_END = '2026-02-04'

MODEL_DIR = 'models'
MODEL_FILE = 'ionis_v2_solar.pth'
PHASE = 'Phase 5: Solar Fidelity (15 features, SFI + Kp)'

# Band IDs after Clean Slate re-ingest (v2.1.0, ADIF standard)
HF_BANDS = list(range(102, 112))  # 102=160m through 111=10m
ROWS_PER_BAND = SAMPLE_SIZE // len(HF_BANDS)  # 1M per band

BAND_TO_HZ = {
    102:  1_836_600,   # 160m
    103:  3_568_600,   # 80m
    104:  5_287_200,   # 60m
    105:  7_038_600,   # 40m
    106: 10_138_700,   # 30m
    107: 14_097_100,   # 20m
    108: 18_104_600,   # 17m
    109: 21_094_600,   # 15m
    110: 24_924_600,   # 12m
    111: 28_124_600,   # 10m
}

FEATURES = [
    'distance', 'freq_log', 'hour_sin', 'hour_cos', 'ssn',
    'az_sin', 'az_cos', 'lat_diff', 'midpoint_lat',
    'season_sin', 'season_cos',
    'ssn_lat_interact', 'day_night_est',
    'prop_quality', 'band_sfi_interact',
]

# Device selection: MPS (M3 Ultra) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ionis-v2')


# ═══════════════════════════════════════════════════════════════════
# 2. Model Architecture — ResNet-style regression (~268k params)
# ═══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class IONIS_V2(nn.Module):
    """Residual Neural Network for SNR regression.

    Architecture: Linear → ReLU → ResBlock × 2 → Linear(1)
    Parameters:   ~269,057 (input=15, hidden=256)
    """
    def __init__(self, input_dim=15, hidden_dim=256):
        super().__init__()
        self.pre = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.post = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.pre(x))
        x = self.res1(x)
        x = self.res2(x)
        return self.post(x)


# ═══════════════════════════════════════════════════════════════════
# 3. Maidenhead Grid Utilities (matches clickhouse_loader.cpp:27-77)
# ═══════════════════════════════════════════════════════════════════

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def clean_grids(raw_grids):
    """Clean FixedString(8) grids from ClickHouse. Invalid grids → 'JJ00'."""
    out = []
    for g in raw_grids:
        if isinstance(g, (bytes, bytearray)):
            s = g.decode('ascii', errors='ignore').rstrip('\x00')
        else:
            s = str(g).rstrip('\x00')
        m = GRID_RE.search(s)
        out.append(m.group(0).upper() if m else 'JJ00')
    return out


def grid_to_latlon(grids):
    """Vectorized Maidenhead 4-char grid → (lat, lon) arrays."""
    grid4 = np.array(grids, dtype='U4')
    codes = grid4.view('U1').reshape(-1, 4)
    b = codes.view(np.uint32).astype(np.float32)
    lon = (b[:, 0] - ord('A')) * 20.0 - 180.0 + (b[:, 2] - ord('0')) * 2.0 + 1.0
    lat = (b[:, 1] - ord('A')) * 10.0 - 90.0 + (b[:, 3] - ord('0')) * 1.0 + 0.5
    return lat, lon


# ═══════════════════════════════════════════════════════════════════
# 4. Feature Engineering (15 normalized features)
# ═══════════════════════════════════════════════════════════════════

def band_to_hz(band_ids):
    """Map band ID array → frequency in Hz. Unknown bands default to 20m."""
    default = 14_097_100
    return np.array([BAND_TO_HZ.get(int(b), default) for b in band_ids],
                    dtype=np.float64)


def engineer_features(distance, freq_hz, hour, month, azimuth,
                      ssn_arr, sfi_arr, kp_arr,
                      tx_lat, tx_lon, rx_lat, rx_lon):
    """Compute 15 normalized features from raw columns. Returns (N, 15) float32.

    Features 1-13:  original IONIS V2 (path geometry, solar, interactions)
    Feature 14:     prop_quality = SFI_norm × (1 - Kp_norm)
    Feature 15:     band_sfi_interact = SFI_norm × freq_log
    """
    f_distance     = distance / 20000.0
    f_freq_log     = np.log10(freq_hz.astype(np.float32)) / 8.0
    f_hour_sin     = np.sin(2.0 * np.pi * hour / 24.0)
    f_hour_cos     = np.cos(2.0 * np.pi * hour / 24.0)
    f_ssn          = ssn_arr / 300.0
    f_az_sin       = np.sin(2.0 * np.pi * azimuth / 360.0)
    f_az_cos       = np.cos(2.0 * np.pi * azimuth / 360.0)
    f_lat_diff     = np.abs(tx_lat - rx_lat) / 180.0
    f_midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    f_season_sin   = np.sin(2.0 * np.pi * month / 12.0)
    f_season_cos   = np.cos(2.0 * np.pi * month / 12.0)
    f_ssn_lat      = f_ssn * np.abs(f_midpoint_lat)
    midpoint_lon   = (tx_lon + rx_lon) / 2.0
    local_solar_h  = hour + midpoint_lon / 15.0
    f_daynight     = np.cos(2.0 * np.pi * local_solar_h / 24.0)

    # Phase 5: Solar fidelity features
    sfi_norm          = sfi_arr / 300.0      # F10.7 solar flux, range ~60-300+
    kp_norm           = kp_arr / 9.0         # Kp geomagnetic index, range 0-9
    f_prop_quality    = sfi_norm * (1.0 - kp_norm)    # high SFI + low Kp = good prop
    f_band_sfi        = sfi_norm * f_freq_log          # SFI effect varies by band

    return np.column_stack([
        f_distance, f_freq_log, f_hour_sin, f_hour_cos, f_ssn,
        f_az_sin, f_az_cos, f_lat_diff, f_midpoint_lat,
        f_season_sin, f_season_cos, f_ssn_lat, f_daynight,
        f_prop_quality, f_band_sfi,
    ]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 5. Streaming Data Ingestion
# ═══════════════════════════════════════════════════════════════════

def build_band_query(band_id):
    """Build a single-band sampling query with 3-hourly Kp + daily SFI/SSN.

    The JOIN uses intDiv(toHour, 3) to align WSPR spots with the matching
    3-hour Kp bucket from solar.indices_raw (GFZ Potsdam format).
    SSN and SFI are daily values replicated across all 8 buckets.
    """
    return f"""
        SELECT s.snr, s.distance, s.band,
               toHour(s.timestamp) AS hour, toMonth(s.timestamp) AS month,
               s.azimuth,
               toString(s.grid) AS grid,
               toString(s.reporter_grid) AS reporter_grid,
               sol.ssn, sol.sfi, sol.kp
        FROM wspr.spots_raw s
        INNER JOIN (
            SELECT date,
                   intDiv(toHour(time), 3) AS bucket,
                   max(ssn) AS ssn,
                   max(observed_flux) AS sfi,
                   max(kp_index) AS kp
            FROM solar.indices_raw FINAL
            GROUP BY date, bucket
            HAVING ssn > 0
        ) sol ON toDate(s.timestamp) = sol.date
             AND intDiv(toHour(s.timestamp), 3) = sol.bucket
        WHERE s.band = {band_id}
          AND s.timestamp >= '{DATE_START}' AND s.timestamp < '{DATE_END}'
          AND s.snr BETWEEN -35 AND 25
          AND s.distance BETWEEN 500 AND 18000
          AND length(s.grid) >= 4 AND length(s.reporter_grid) >= 4
        ORDER BY cityHash64(toString(s.timestamp))
        LIMIT {ROWS_PER_BAND}
    """


def build_fallback_query():
    """Non-stratified fallback query with 3-hourly Kp + daily SFI/SSN."""
    band_list = ','.join(str(b) for b in HF_BANDS)
    return f"""
        WITH solar_bucketed AS (
            SELECT date,
                   intDiv(toHour(time), 3) AS bucket,
                   max(ssn) AS ssn,
                   max(observed_flux) AS sfi,
                   max(kp_index) AS kp
            FROM solar.indices_raw FINAL
            GROUP BY date, bucket
            HAVING ssn > 0
        )
        SELECT s.snr, s.distance, s.band,
               toHour(s.timestamp), toMonth(s.timestamp), s.azimuth,
               toString(s.grid), toString(s.reporter_grid),
               sol.ssn, sol.sfi, sol.kp
        FROM wspr.spots_raw s
        INNER JOIN solar_bucketed sol ON toDate(s.timestamp) = sol.date
             AND intDiv(toHour(s.timestamp), 3) = sol.bucket
        WHERE s.timestamp >= '{DATE_START}' AND s.timestamp < '{DATE_END}'
          AND s.snr BETWEEN -35 AND 25
          AND s.band IN ({band_list})
          AND s.distance BETWEEN 500 AND 18000
          AND length(s.grid) >= 4 AND length(s.reporter_grid) >= 4
        LIMIT {SAMPLE_SIZE}
    """


def connect_clickhouse():
    """Connect to ClickHouse with exponential-backoff retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = clickhouse_connect.get_client(
                host=CH_HOST,
                port=CH_PORT,
                connect_timeout=CH_CONNECT_TIMEOUT,
                send_receive_timeout=CH_QUERY_TIMEOUT,
            )
            ver = client.server_version
            log.info(f"Connected to ClickHouse {ver} at {CH_HOST}:{CH_PORT}")
            return client
        except Exception as e:
            log.warning(f"Connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise ConnectionError(
                    f"Cannot reach ClickHouse at {CH_HOST}:{CH_PORT} "
                    f"after {MAX_RETRIES} attempts"
                ) from e
            time.sleep(2 ** attempt)


def process_raw_columns(snr, distance, band_id, hour, month,
                        azimuth, raw_tx, raw_rx, ssn_arr, sfi_arr, kp_arr):
    """Convert raw ClickHouse columns → (features, targets) numpy arrays."""
    freq_hz = band_to_hz(band_id)
    tx_grids = clean_grids(raw_tx)
    rx_grids = clean_grids(raw_rx)
    tx_lat, tx_lon = grid_to_latlon(tx_grids)
    rx_lat, rx_lon = grid_to_latlon(rx_grids)
    feats = engineer_features(
        distance, freq_hz, hour, month, azimuth,
        ssn_arr, sfi_arr, kp_arr, tx_lat, tx_lon, rx_lat, rx_lon,
    )
    return feats, snr


def stream_dataset():
    """Sequential per-band ingestion from ClickHouse → (X, y) tensors.

    Queries one band at a time (102, 103, ..., 111) so ClickHouse only
    sorts and streams 1M rows per query instead of 10M in a single
    UNION ALL. Results are concatenated on the M3 Ultra (96 GB unified).
    """
    client = connect_clickhouse()
    band_range = f"{HF_BANDS[0]}-{HF_BANDS[-1]}"

    log.info(f"Sequential sampling {DATE_START} to {DATE_END}, HF bands {band_range}")
    log.info(f"  {ROWS_PER_BAND:,}/band x {len(HF_BANDS)} bands = {SAMPLE_SIZE:,}")
    log.info(f"  CH settings: external_group_by=20G, max_memory=80G, max_threads=16")

    t0 = time.perf_counter()
    feat_blocks = []
    tgt_blocks = []
    band_counts = {}
    ssn_accum = []
    snr_accum = []
    total_rows = 0

    for band_idx, bid in enumerate(HF_BANDS, 1):
        hz = BAND_TO_HZ.get(bid, 0)
        label = f"{hz / 1e6:.3f}MHz" if hz else f"band={bid}"
        log.info(f"  Band {bid} ({label}) [{band_idx}/{len(HF_BANDS)}]...")

        query = build_band_query(bid)
        t_band = time.perf_counter()
        band_rows = 0

        try:
            with client.query_column_block_stream(
                query, settings=CH_SETTINGS,
            ) as stream:
                for block in stream:
                    snr      = np.asarray(block[0], dtype=np.float32)
                    distance = np.asarray(block[1], dtype=np.float32)
                    band_id  = np.asarray(block[2], dtype=np.int32)
                    hour     = np.asarray(block[3], dtype=np.float32)
                    month    = np.asarray(block[4], dtype=np.float32)
                    azimuth  = np.asarray(block[5], dtype=np.float32)
                    raw_tx   = list(block[6])
                    raw_rx   = list(block[7])
                    ssn_arr  = np.asarray(block[8], dtype=np.float32)
                    sfi_arr  = np.asarray(block[9], dtype=np.float32)
                    kp_arr   = np.asarray(block[10], dtype=np.float32)

                    n = len(snr)
                    band_rows += n
                    total_rows += n

                    ssn_accum.append(ssn_arr)
                    snr_accum.append(snr)

                    feats, tgts = process_raw_columns(
                        snr, distance, band_id, hour, month,
                        azimuth, raw_tx, raw_rx, ssn_arr, sfi_arr, kp_arr,
                    )
                    feat_blocks.append(feats)
                    tgt_blocks.append(tgts)

        except Exception as e:
            log.warning(f"    Band {bid} stream error after {band_rows:,} rows: {e}")
            if total_rows == 0:
                raise

        band_sec = time.perf_counter() - t_band
        band_rps = band_rows / band_sec if band_sec > 0 else 0
        band_counts[bid] = band_rows
        log.info(
            f"    {band_rows:>10,} rows in {band_sec:5.1f}s "
            f"({band_rps:,.0f} rows/sec) | cumulative: {total_rows:,}"
        )

    elapsed = time.perf_counter() - t0
    rps = total_rows / elapsed if elapsed > 0 else 0
    log.info(f"Ingestion complete: {total_rows:,} rows in {elapsed:.1f}s ({rps:,.0f} rows/sec)")

    if total_rows == 0:
        raise RuntimeError("No rows ingested — check ClickHouse filters and solar data")

    X_np = np.concatenate(feat_blocks, axis=0)
    y_np = np.concatenate(tgt_blocks, axis=0)
    ssn_all = np.concatenate(ssn_accum, axis=0)
    snr_all = np.concatenate(snr_accum, axis=0)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    return (X, y), band_counts, ssn_all, snr_all


def _fallback_ingest(client):
    """Non-streamed fallback if column block stream fails."""
    query = build_fallback_query()
    log.info("Running fallback query (non-streamed)...")
    t0 = time.perf_counter()
    result = client.query(query, settings=CH_SETTINGS)
    rows = result.result_rows
    elapsed = time.perf_counter() - t0
    n = len(rows)
    log.info(f"Fallback: {n:,} rows in {elapsed:.1f}s")

    if n == 0:
        raise RuntimeError("Fallback query returned 0 rows")

    cols = list(zip(*rows))
    snr      = np.array(cols[0], dtype=np.float32)
    distance = np.array(cols[1], dtype=np.float32)
    band_id  = np.array(cols[2], dtype=np.int32)
    hour     = np.array(cols[3], dtype=np.float32)
    month    = np.array(cols[4], dtype=np.float32)
    azimuth  = np.array(cols[5], dtype=np.float32)
    raw_tx   = list(cols[6])
    raw_rx   = list(cols[7])
    ssn_arr  = np.array(cols[8], dtype=np.float32)
    sfi_arr  = np.array(cols[9], dtype=np.float32)
    kp_arr   = np.array(cols[10], dtype=np.float32)

    feats, tgts = process_raw_columns(
        snr, distance, band_id, hour, month,
        azimuth, raw_tx, raw_rx, ssn_arr, sfi_arr, kp_arr,
    )
    X = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(tgts, dtype=torch.float32).unsqueeze(1)
    return X, y


# ═══════════════════════════════════════════════════════════════════
# 6. Metrics
# ═══════════════════════════════════════════════════════════════════

def pearson_r(pred, target):
    """Pearson correlation between two 1-D tensors."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# 7. Training Loop
# ═══════════════════════════════════════════════════════════════════

def print_diagnostics(X, y, band_counts, ssn_all, snr_all):
    """Print dataset diagnostics: feature stats, band distribution, correlations."""
    n = len(X)
    snr_mean = y.mean().item()
    snr_std = y.std().item()

    log.info(f"Dataset: {n:,} rows x {INPUT_DIM} features")
    log.info(f"SNR range: {y.min().item():.0f} to {y.max().item():.0f} dB")
    log.info(f"SNR mean: {snr_mean:.1f} dB, std: {snr_std:.1f} dB")

    # Feature statistics
    log.info("Feature statistics (normalized):")
    log.info(f"  {'Feature':<20s}  {'Min':>8s}  {'Mean':>8s}  {'Max':>8s}")
    log.info(f"  {'-' * 50}")
    for i, name in enumerate(FEATURES):
        col = X[:, i]
        log.info(f"  {name:<20s}  {col.min():8.4f}  {col.mean():8.4f}  {col.max():8.4f}")

    # Per-band row counts
    if band_counts:
        log.info("Per-band row counts (stratified):")
        for bid in sorted(band_counts):
            hz = BAND_TO_HZ.get(bid, 0)
            label = f"{hz / 1e6:.3f}MHz" if hz else f"band={bid}"
            log.info(f"  Band {bid:3d} ({label:>10s}): {band_counts[bid]:>10,} rows")

    # SSN-SNR correlation diagnostic
    if len(ssn_all) > 0 and len(snr_all) > 0:
        corr = np.corrcoef(ssn_all, snr_all)[0, 1]
        log.info(f"SSN-SNR Pearson correlation: {corr:+.4f}")
        if corr > 0.1:
            log.info("  TARGET MET: correlation > 0.1")
        elif corr > 0:
            log.info("  Positive but below 0.1 target")
        else:
            log.info("  WARNING: Negative correlation")

    return snr_mean, snr_std


def main():
    log.info(f"IONIS V2 Oracle | {PHASE}")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    log.info(f"Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")

    # ── Stream data from ClickHouse ──
    (X, y), band_counts, ssn_all, snr_all = stream_dataset()
    snr_mean, snr_std = print_diagnostics(X, y, band_counts, ssn_all, snr_all)
    n = len(X)

    # ── Train/val split ──
    dataset = TensorDataset(X, y)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model ──
    model = IONIS_V2(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {INPUT_DIM} -> {HIDDEN_DIM} -> 1  ({params:,} parameters)")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    best_val_loss = float('inf')

    # ── Training ──
    log.info("Training started")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  {'LR':>10s}  {'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.perf_counter()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

        # Validate (accumulate for Pearson)
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by)
                val_loss_sum += loss.item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        val_pearson = pearson_r(preds_cat, targets_cat)

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        epoch_sec = time.perf_counter() - t_epoch

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'snr_mean': snr_mean,
                'snr_std': snr_std,
                'input_dim': INPUT_DIM,
                'hidden_dim': HIDDEN_DIM,
                'features': FEATURES,
                'band_ids': HF_BANDS,
                'band_to_hz': BAND_TO_HZ,
                'date_range': f'{DATE_START} to {DATE_END}',
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'phase': PHASE,
                'solar_resolution': '3-hourly Kp, daily SFI/SSN (GFZ Potsdam)',
            }, model_path)
            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.2f}dB  {val_pearson:+7.4f}  "
            f"{lr_now:.2e}  {epoch_sec:5.1f}s{marker}"
        )

    best_rmse = np.sqrt(best_val_loss)
    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.2f} dB")
    log.info(f"Checkpoint: {model_path}")


if __name__ == '__main__':
    main()
