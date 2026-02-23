#!/usr/bin/env python3
"""
train_v22.py — IONIS V22 Training

V22 adds:
    - Solar elevation angles (tx/rx) for precise D-layer/F-layer boundaries
    - Band×darkness cross-products for band-specific time penalties
    - Replaces V21-beta's sigmoid-based physics gates

Key insight: V21-beta learned WHEN bands peak but not HOW MUCH they close.
Cross-products (freq_log × solar_elevation) give the model mathematical
permission to apply massive penalties to high bands at night without
affecting low bands.

All configuration loaded from config_v22.json.
Architecture, features, dataset, and data loading from common/train_common.py.

Architecture constraints (non-negotiable — V16 Physics Laws):
    1. Architecture: IonisGate (context-aware gates from trunk output)
    2. Loss: HuberLoss(delta=1.0) — robust to synthetic anchors
    3. Regularization: Gate variance loss — forces context sensitivity
    4. Init: Defibrillator — weights uniform(0.8-1.2), fc2.bias=-10
    5. Constraint: Weight clamp [0.5, 2.0] after EVERY step

Success criteria: TST-900 >= 8/10, Pearson >= +0.46, SFI > +0.4σ
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
sys.path.insert(0, VERSIONS_DIR)
sys.path.insert(0, COMMON_DIR)  # For model.py imports within train_common.py

from common.train_common import (
    IonisGate,
    SignatureDataset,
    engineer_features,
    init_defibrillator,
    clamp_sidecars,
    get_optimizer_groups,
    load_source_data,
    log_config,
)

# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v22.json")

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
        config_json = json.dumps(config).replace('"', '\\"')
        features_array = "['" + "','".join(config["features"]) + "']"
        query = f"""
            INSERT INTO training.runs (
                run_id, version, variant, status, hostname,
                started_at, config_json,
                dnn_dim, hidden_dim, input_dim, training_epochs, batch_size,
                trunk_lr, sidecar_lr,
                wspr_sample, rbn_full_sample, rbn_dx_upsample, contest_upsample,
                features
            ) VALUES (
                '{run_id}', '{config["version"]}', '{config.get("variant", "alpha")}', 'running', '{socket.gethostname()}',
                now(),
                '{config_json}',
                {config["model"]["dnn_dim"]}, {config["model"]["hidden_dim"]},
                {config["model"]["input_dim"]}, {config["training"]["epochs"]},
                {config["training"]["batch_size"]},
                {config["training"]["trunk_lr"]}, {config["training"]["sidecar_lr"]},
                {config["data"]["wspr_sample"]}, {config["data"].get("rbn_full_sample", 0)},
                {config["data"]["rbn_dx_upsample"]}, {config["data"]["contest_upsample"]},
                {features_array}
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
        pass  # Silent fail on epoch audit


def audit_complete(run_id: str, config: dict, train_rows: int, val_rows: int,
                   date_range: str, val_rmse: float, val_pearson: float,
                   sfi_benefit: float, storm_cost: float, best_epoch: int,
                   wall_seconds: float):
    """Record training completion in ClickHouse (supersedes start via RBMT)."""
    try:
        config_json = json.dumps(config).replace('"', '\\"')
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
                val_rmse, val_pearson, sfi_benefit, storm_cost, best_epoch
            ) VALUES (
                '{run_id}', '{config["version"]}', '{config.get("variant", "alpha")}', 'completed', '{socket.gethostname()}',
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
                {val_rmse}, {val_pearson}, {sfi_benefit}, {storm_cost}, {best_epoch}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit complete failed: {e}")


# ── Per-Band Normalization (per-source per-band) ─────────────────────────────

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

    log_config(CONFIG, CONFIG_FILE, DEVICE)
    audit_start(run_id, CONFIG)

    # ── Load Data (with day_of_year for V22) ──
    ch = CONFIG["clickhouse"]
    client = clickhouse_connect.get_client(host=ch["host"], port=ch["port"])

    # V22: Include day_of_year for solar elevation computation
    wspr_df = load_source_data(client, "wspr.signatures_v2_terrestrial",
                               CONFIG["data"]["wspr_sample"],
                               include_day_of_year=True)
    rbn_dx_df = load_source_data(client, "rbn.dxpedition_signatures", None,
                                  include_day_of_year=True)
    contest_df = load_source_data(client, "contest.signatures", None,
                                   include_day_of_year=True)

    # Optional RBN Full (Pressure Vessel experiment)
    rbn_full_sample = CONFIG["data"].get("rbn_full_sample", 0)
    if rbn_full_sample > 0:
        log.info(f"Loading RBN Full (Pressure Vessel): {rbn_full_sample:,} rows")
        rbn_full_df = load_source_data(client, "rbn.signatures", rbn_full_sample,
                                        include_day_of_year=True)
    else:
        rbn_full_df = None

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

    if rbn_full_df is not None:
        rbn_full_snr = normalize_snr_per_band(rbn_full_df, 'rbn', norm_consts)
        log.info(f"  RBN Full: mean={rbn_full_snr.mean():.3f}, std={rbn_full_snr.std():.3f}")

    # ── Features (V22: solar elevation + cross-products) ──
    log.info("")
    log.info("Engineering features...")
    log.info(f"  dnn_dim={CONFIG['model']['dnn_dim']} (solar_dep + freq×dark cross-products)")

    wspr_X = engineer_features(wspr_df, CONFIG)
    rbn_dx_X = engineer_features(rbn_dx_df, CONFIG)
    contest_X = engineer_features(contest_df, CONFIG)
    if rbn_full_df is not None:
        rbn_full_X = engineer_features(rbn_full_df, CONFIG)

    # ── Weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w /= wspr_w.mean()
    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w /= rbn_dx_w.mean()
    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w /= contest_w.mean()
    if rbn_full_df is not None:
        rbn_full_w = rbn_full_df['spot_count'].values.astype(np.float32)
        rbn_full_w /= rbn_full_w.mean()

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
    if rbn_full_df is not None:
        X = np.vstack([wspr_X, rbn_dx_X_up, contest_X_up, rbn_full_X])
        y = np.concatenate([wspr_snr, rbn_dx_snr_up, contest_snr_up, rbn_full_snr]).reshape(-1, 1)
        w = np.concatenate([wspr_w, rbn_dx_w_up, contest_w_up, rbn_full_w]).reshape(-1, 1)
    else:
        X = np.vstack([wspr_X, rbn_dx_X_up, contest_X_up])
        y = np.concatenate([wspr_snr, rbn_dx_snr_up, contest_snr_up]).reshape(-1, 1)
        w = np.concatenate([wspr_w, rbn_dx_w_up, contest_w_up]).reshape(-1, 1)

    n = len(X)
    log.info("")
    log.info(f"Combined dataset: {n:,} rows")
    log.info(f"  WSPR:     {len(wspr_X):,} ({100*len(wspr_X)/n:.1f}%)")
    log.info(f"  RBN DX:   {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/n:.1f}%)")
    log.info(f"  Contest:  {len(contest_snr_up):,} ({100*len(contest_snr_up)/n:.1f}%)")
    if rbn_full_df is not None:
        log.info(f"  RBN Full: {len(rbn_full_X):,} ({100*len(rbn_full_X)/n:.1f}%)")
    log.info(f"Normalized SNR: mean={y.mean():.3f}, std={y.std():.3f}")
    log.info(f"Source data range: {date_range}")

    del wspr_df, rbn_dx_df, contest_df
    del wspr_X, rbn_dx_X, contest_X, rbn_dx_X_up, contest_X_up
    if rbn_full_df is not None:
        del rbn_full_df, rbn_full_X
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

    # ── Model (Constraint #1) ──
    model = IonisGate(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        sfi_idx=CONFIG["model"]["sfi_idx"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisGate ({total_params:,} params)")

    # ── Defibrillator (Constraint #4) ──
    init_defibrillator(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

    # ── Optimizer (6-group) ──
    param_groups = get_optimizer_groups(
        model,
        trunk_lr=CONFIG["training"]["trunk_lr"],
        scaler_lr=CONFIG["training"]["scaler_lr"],
        sidecar_lr=CONFIG["training"]["sidecar_lr"],
    )
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG["training"]["weight_decay"])

    epochs = CONFIG["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── HuberLoss (Constraint #2) ──
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
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  {'Time':>6s}")
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

            # Constraint #3: Gate variance loss
            out, sun_gate, storm_gate = model.forward_with_gates(bx)
            primary_loss = (criterion(out, by) * bw).mean()
            var_loss = -lambda_var * (sun_gate.var() + storm_gate.var())
            loss = primary_loss + var_loss

            loss.backward()
            optimizer.step()

            # Constraint #5: Weight clamp
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

            # Save using safetensors (no pickle)
            save_safetensors(model.state_dict(), model_path)

            # Save metadata as companion JSON
            metadata = {
                'version': CONFIG['version'],
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_benefit': sfi_benefit,
                'storm_cost': storm_cost,
                'rbn_full_sample': rbn_full_sample,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.2f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        # Audit trail: record epoch metrics
        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    sfi_benefit, storm_cost, epoch_sec, is_best)

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    wall_seconds = time.perf_counter() - training_start_time

    # Audit trail: record completion (supersedes start via ReplacingMergeTree)
    audit_complete(run_id, CONFIG, train_size, val_size, date_range,
                   best_rmse, best_pearson, best_sfi, best_kp, best_epoch, wall_seconds)

    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    log.info("")
    log.info("=" * 70)
    log.info("V22 RESULTS")
    log.info("=" * 70)

    p_min = CONFIG["validation"]["pearson_min"]
    kp_min = CONFIG["validation"]["kp_storm_min"]
    sfi_min = CONFIG["validation"]["sfi_benefit_min"]

    p_pass = best_pearson >= p_min
    kp_pass = best_kp >= kp_min
    sfi_pass = best_sfi >= sfi_min

    log.info("")
    log.info("SUCCESS CRITERIA:")
    log.info(f"  Pearson >= +{p_min}:    {best_pearson:+.4f}  {'PASS' if p_pass else 'FAIL'}")
    log.info(f"  Kp sidecar >= +{kp_min}σ: {best_kp:+.3f}σ  {'PASS' if kp_pass else 'FAIL'}")
    log.info(f"  SFI sidecar >= +{sfi_min}σ: {best_sfi:+.3f}σ  {'PASS' if sfi_pass else 'FAIL'}")

    if p_pass and sfi_pass:
        log.info("")
        log.info(">>> V22 TRAINING: SUCCESS <<<")
        log.info(">>> Run TST-900 to verify band×time discrimination <<<")
    else:
        log.info("")
        log.info(">>> V22 TRAINING: NEEDS REVIEW <<<")
        log.info(">>> Check training logs <<<")

    log.info("")
    log.info("V21-beta REFERENCE: Pearson +0.464, Kp +1.29σ, SFI +0.48σ, TST-900 4/10")
    log.info("V22 TARGET: TST-900 >= 8/10 (band×time discrimination)")


if __name__ == '__main__':
    train()
