#!/usr/bin/env python3
"""
train_audit01.py — AUDIT-01: Widen Clamp [0.5, 2.0] -> [0.1, 5.0]

HYPOTHESIS: SFI sidecar +0.48 sigma is the clamp floor, not physics.
            Widening the clamp should cause SFI to drop below +0.48.

RECIPE: V22-gamma (dnn_dim=15, no IRI features)
CHANGE: Weight clamp only [0.1, 5.0]
DURATION: 30 epochs

INSTRUMENTATION: Log weight min/max for sun sidecar fc1/fc2 every epoch.
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
sys.path.insert(0, COMMON_DIR)

from common.train_common import (
    IonisGate,
    SignatureDataset,
    engineer_features,
    init_defibrillator,
    get_optimizer_groups,
    load_source_data,
    log_config,
)

# ── AUDIT-01: WIDENED CLAMP ─────────────────────────────────────────────────
# This is the ONLY change from V22-gamma

CLAMP_MIN = 0.1  # Was 0.5
CLAMP_MAX = 5.0  # Was 2.0


def clamp_sidecars_wide(model):
    """
    Clamp sidecar weights to [0.1, 5.0] (widened from [0.5, 2.0]).

    AUDIT-01: Testing if the +0.48 fixed point was the clamp floor.
    """
    with torch.no_grad():
        for sidecar in [model.sun_sidecar, model.storm_sidecar]:
            sidecar.fc1.weight.clamp_(CLAMP_MIN, CLAMP_MAX)
            sidecar.fc2.weight.clamp_(CLAMP_MIN, CLAMP_MAX)


def log_sidecar_weights(model, epoch, log):
    """Log sun sidecar weight statistics for auditing."""
    with torch.no_grad():
        fc1_w = model.sun_sidecar.fc1.weight.cpu().numpy()
        fc2_w = model.sun_sidecar.fc2.weight.cpu().numpy()
        log.info(f"  SUN SIDECAR WEIGHTS (epoch {epoch}):")
        log.info(f"    fc1: min={fc1_w.min():.4f}, max={fc1_w.max():.4f}, mean={fc1_w.mean():.4f}")
        log.info(f"    fc2: min={fc2_w.min():.4f}, max={fc2_w.max():.4f}, mean={fc2_w.mean():.4f}")

        # Also log storm sidecar for comparison
        storm_fc1 = model.storm_sidecar.fc1.weight.cpu().numpy()
        storm_fc2 = model.storm_sidecar.fc2.weight.cpu().numpy()
        log.info(f"  STORM SIDECAR WEIGHTS (epoch {epoch}):")
        log.info(f"    fc1: min={storm_fc1.min():.4f}, max={storm_fc1.max():.4f}")
        log.info(f"    fc2: min={storm_fc2.min():.4f}, max={storm_fc2.max():.4f}")


# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_audit01.json")

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
                'AUDIT-01: Widen clamp [0.1, 5.0]'
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit start failed: {e}")


def audit_epoch(run_id: str, epoch: int, train_loss: float, val_loss: float,
                val_rmse: float, val_pearson: float, sfi_benefit: float,
                storm_cost: float, epoch_seconds: float, is_best: bool,
                sun_fc1_min: float = None, sun_fc1_max: float = None,
                sun_fc2_min: float = None, sun_fc2_max: float = None):
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
                'AUDIT-01: Widen clamp [0.1, 5.0] - COMPLETED'
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
    log.info("AUDIT-01: WIDEN CLAMP [0.5, 2.0] -> [0.1, 5.0]")
    log.info("=" * 70)
    log.info("")
    log.info("HYPOTHESIS: SFI +0.48 is the clamp floor, not physics.")
    log.info("EXPECTED: SFI drops below +0.48 when floor is removed.")
    log.info("")

    log_config(CONFIG, CONFIG_FILE, DEVICE)
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

    # ── Features (V22-gamma: solar elevation + cross-products) ──
    log.info("")
    log.info("Engineering features...")
    log.info(f"  dnn_dim={CONFIG['model']['dnn_dim']} (V22-gamma recipe)")

    wspr_X = engineer_features(wspr_df, CONFIG)
    rbn_dx_X = engineer_features(rbn_dx_df, CONFIG)
    contest_X = engineer_features(contest_df, CONFIG)

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

    # ── Defibrillator ──
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
    log.info(f"*** USING WIDENED CLAMP [{CLAMP_MIN}, {CLAMP_MAX}] ***")
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

            # AUDIT-01: WIDENED CLAMP
            clamp_sidecars_wide(model)

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
                'audit': 'AUDIT-01: Widen clamp [0.1, 5.0]',
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_benefit': sfi_benefit,
                'storm_cost': storm_cost,
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

        # Log weights every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            log_sidecar_weights(model, epoch, log)

        # Get weight stats for audit
        with torch.no_grad():
            sun_fc1_min = model.sun_sidecar.fc1.weight.min().item()
            sun_fc1_max = model.sun_sidecar.fc1.weight.max().item()
            sun_fc2_min = model.sun_sidecar.fc2.weight.min().item()
            sun_fc2_max = model.sun_sidecar.fc2.weight.max().item()

        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    sfi_benefit, storm_cost, epoch_sec, is_best,
                    sun_fc1_min, sun_fc1_max, sun_fc2_min, sun_fc2_max)

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    wall_seconds = time.perf_counter() - training_start_time

    audit_complete(run_id, CONFIG, train_size, val_size, date_range,
                   best_rmse, best_pearson, best_sfi, best_kp, best_epoch, wall_seconds)

    log.info("-" * len(hdr))
    log.info("")
    log.info("=" * 70)
    log.info("AUDIT-01 RESULTS")
    log.info("=" * 70)

    log.info("")
    log.info("=== FINAL WEIGHTS ===")
    log_sidecar_weights(model, epochs, log)

    log.info("")
    log.info("AUDIT-01 SUMMARY:")
    log.info(f"  Best RMSE: {best_rmse:.4f}σ")
    log.info(f"  Best Pearson: {best_pearson:+.4f}")
    log.info(f"  Final SFI+: {best_sfi:+.4f}σ")
    log.info(f"  Final Kp9-: {best_kp:+.4f}σ")

    log.info("")
    if best_sfi < 0.48:
        log.info(">>> HYPOTHESIS CONFIRMED <<<")
        log.info(f">>> SFI dropped to {best_sfi:+.4f}σ (below +0.48) <<<")
        log.info(">>> The +0.48 was the clamp floor, not physics <<<")
    else:
        log.info(">>> HYPOTHESIS NOT CONFIRMED <<<")
        log.info(f">>> SFI is still at {best_sfi:+.4f}σ <<<")
        log.info(">>> Need further investigation <<<")

    log.info("")
    log.info(f"Checkpoint: {model_path}")
    log.info(f"Wall time: {wall_seconds/60:.1f} minutes")


if __name__ == '__main__':
    train()
