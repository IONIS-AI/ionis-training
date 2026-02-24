#!/usr/bin/env python3
"""
train_v25_beta.py — IONIS V25-beta Training

V25-beta: Force the SFI×freq interaction by MULTIPLYING before the sidecar.

The single change from V22-gamma:
    - Sun sidecar input: SFI (1D) → SFI * freq_log (1D product)

The optimizer cannot separate SFI from freq because they enter as a product.
High band + high SFI = big number → big boost
Low band + high SFI = smaller number → smaller boost

WHAT DOES NOT CHANGE (identical to V22-gamma):
    - Trunk: 15 → 512 → 256 → 128 → 1
    - Storm sidecar: Kp → 8 → 1 (1D input, unchanged)
    - Clamp: [0.5, 2.0] both sidecars
    - Defibrillator: both sidecars
    - Gates: both gates, gate variance loss
    - Features: 15D V22-gamma recipe (no IRI)
    - Data recipe: 20M WSPR + 13M DX (50x) + 5.7M Contest
    - HuberLoss(delta=1.0)
    - All learning rates, weight decay, batch size, epoch count
"""

import gc
import json
import logging
import math
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
    SignatureDataset,
    engineer_features,
    load_source_data,
    log_config,
    grid4_to_latlon_arrays,
)

# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v25_beta.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── V25-beta Model: MonotonicMLP with SFI*freq_log product input ─────────────

class MonotonicMLP(nn.Module):
    """Monotonically increasing MLP for physics constraints."""

    def __init__(self, hidden_dim=8, input_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(nn.functional.linear(x, w1, self.fc1.bias))
        return nn.functional.linear(h, w2, self.fc2.bias)


def _gate(x):
    """Gate function: range 0.5 to 2.0"""
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisGateV25Beta(nn.Module):
    """
    IONIS V25-beta — Sun sidecar receives SFI * freq_log as 1D input.
    
    The interaction is FORCED by multiplication before entering the sidecar.
    The optimizer cannot separate SFI from freq_log.
    """

    def __init__(self, dnn_dim=15, sidecar_hidden=8, sfi_idx=15, kp_penalty_idx=16,
                 gate_init_bias=None, sun_sidecar_input_dim=1):
        super().__init__()

        if gate_init_bias is None:
            gate_init_bias = -math.log(2.0)

        self.dnn_dim = dnn_dim
        self.sfi_idx = sfi_idx
        self.kp_penalty_idx = kp_penalty_idx

        # Trunk: geography/time features → 256-dim representation
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        # Base head: trunk → SNR prediction
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )

        # Scaler heads: trunk → gate logits
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # V25-beta: Sun sidecar takes 1D input (SFI * freq_log product)
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden, input_dim=1)
        # Storm sidecar unchanged
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden, input_dim=1)

        self._init_scaler_heads(gate_init_bias)

    def _init_scaler_heads(self, gate_init_bias):
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, gate_init_bias)

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        # V25-beta: MULTIPLY SFI and freq_log
        x_freq_log = x[:, 1:2]  # freq_log is at index 1
        x_sun_input = x_sfi * x_freq_log  # [batch, 1] - the forced interaction

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        return base_snr + sun_gate * self.sun_sidecar(x_sun_input) + \
               storm_gate * self.storm_sidecar(x_kp)

    def forward_with_gates(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        # V25-beta: MULTIPLY
        x_freq_log = x[:, 1:2]
        x_sun_input = x_sfi * x_freq_log

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        sun_boost = self.sun_sidecar(x_sun_input)
        storm_boost = self.storm_sidecar(x_kp)

        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost, \
               sun_gate, storm_gate

    def get_sun_effect(self, sfi_freq_product, device):
        """Get raw sun sidecar output for given SFI*freq_log product."""
        with torch.no_grad():
            x = torch.tensor([[sfi_freq_product]], dtype=torch.float32, device=device)
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty, device):
        with torch.no_grad():
            x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
            return self.storm_sidecar(x).item()

    def get_sun_effect_by_band(self, sfi_normalized, device):
        """
        Get sun sidecar output for each band at a given SFI.
        Input is SFI * freq_log product.
        """
        band_freq_hz = {
            "160m": 1_836_600,
            "80m":  3_568_600,
            "40m":  7_038_600,
            "30m": 10_138_700,
            "20m": 14_097_100,
            "15m": 21_094_600,
            "10m": 28_124_600,
        }

        results = {}
        for band, freq_hz in band_freq_hz.items():
            freq_log_norm = np.log10(freq_hz) / 8.0
            product = sfi_normalized * freq_log_norm
            results[band] = self.get_sun_effect(product, device)

        return results


# ── Training Helpers ──────────────────────────────────────────────────────────

def init_defibrillator(model):
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

    model.sun_sidecar.fc1.bias.requires_grad = False
    model.sun_sidecar.fc2.bias.requires_grad = True
    model.storm_sidecar.fc1.bias.requires_grad = False
    model.storm_sidecar.fc2.bias.requires_grad = True

    log.info("Defibrillator applied: sidecar weights uniform(0.8-1.2), fc2.bias=-10.0, fc1.bias frozen")


def clamp_sidecars(model):
    with torch.no_grad():
        for sidecar in [model.sun_sidecar, model.storm_sidecar]:
            sidecar.fc1.weight.clamp_(0.5, 2.0)
            sidecar.fc2.weight.clamp_(0.5, 2.0)


def get_optimizer_groups(model, trunk_lr=1e-5, scaler_lr=5e-5, sidecar_lr=1e-3):
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


# ── Training Audit Trail ─────────────────────────────────────────────────────

AUDIT_HOST = "http://10.60.1.1:8123"

def audit_start(run_id: str, config: dict):
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
                '{run_id}', '{config["version"]}', '{config.get("variant", "beta")}', 'running', '{socket.gethostname()}',
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
                '{run_id}', '{config["version"]}', '{config.get("variant", "beta")}', 'completed', '{socket.gethostname()}',
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


# ── Per-Band Normalization ────────────────────────────────────────────────────

def normalize_snr_per_band(df, source, norm_constants):
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
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    run_id = str(uuid.uuid4())
    training_start_time = time.perf_counter()

    log_config(CONFIG, CONFIG_FILE, DEVICE)

    log.info("")
    log.info("=== V25-BETA: FORCED SFI×FREQ INTERACTION ===")
    log.info("  Sun sidecar input: SFI * freq_log (1D product)")
    log.info("  Optimizer CANNOT separate SFI from frequency")
    log.info("  All other parameters IDENTICAL to V22-gamma")
    log.info("")

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

    # ── Features ──
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

    # ── Upsample DXpedition ──
    rbn_dx_up = CONFIG["data"]["rbn_dx_upsample"]
    log.info(f"Upsampling RBN DXpedition {rbn_dx_up}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (rbn_dx_up, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr, rbn_dx_up)
    rbn_dx_w_up = np.tile(rbn_dx_w, rbn_dx_up)

    # ── Combine ──
    X = np.vstack([wspr_X, rbn_dx_X_up, contest_X])
    y = np.concatenate([wspr_snr, rbn_dx_snr_up, contest_snr]).reshape(-1, 1)
    w = np.concatenate([wspr_w, rbn_dx_w_up, contest_w]).reshape(-1, 1)

    n = len(X)
    log.info("")
    log.info(f"Combined dataset: {n:,} rows")
    log.info(f"  WSPR:     {len(wspr_X):,} ({100*len(wspr_X)/n:.1f}%)")
    log.info(f"  RBN DX:   {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/n:.1f}%)")
    log.info(f"  Contest:  {len(contest_X):,} ({100*len(contest_X)/n:.1f}%)")
    log.info(f"Normalized SNR: mean={y.mean():.3f}, std={y.std():.3f}")
    log.info(f"Source data range: {date_range}")

    del wspr_df, rbn_dx_df, contest_df
    del wspr_X, rbn_dx_X, contest_X, rbn_dx_X_up
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

    # ── Model (V25-beta: SFI * freq_log product) ──
    model = IonisGateV25Beta(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        sfi_idx=CONFIG["model"]["sfi_idx"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
        sun_sidecar_input_dim=CONFIG["model"]["sun_sidecar_input_dim"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisGateV25Beta ({total_params:,} params)")
    log.info(f"  Sun sidecar: SFI * freq_log (1D product input)")
    log.info(f"  Storm sidecar: 1D input (Kp penalty)")

    # ── Defibrillator ──
    init_defibrillator(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

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
    best_sfi_10m = 0.0
    best_sfi_160m = 0.0
    best_epoch = 0

    # ── Training Loop ──
    log.info("")
    log.info(f"Training started ({epochs} epochs)")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI10m':>6s}  {'SFI160m':>7s}  {'Kp9-':>5s}  {'Time':>6s}")
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

        # V25-beta: Get band-specific SFI effects (via product)
        sfi_effects = model.get_sun_effect_by_band(200.0 / 300.0, DEVICE)
        sfi_effects_low = model.get_sun_effect_by_band(70.0 / 300.0, DEVICE)

        sfi_10m = sfi_effects["10m"] - sfi_effects_low["10m"]
        sfi_160m = sfi_effects["160m"] - sfi_effects_low["160m"]

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
            best_sfi_10m = sfi_10m
            best_sfi_160m = sfi_160m
            best_epoch = epoch

            save_safetensors(model.state_dict(), model_path)

            metadata = {
                'version': CONFIG['version'],
                'variant': CONFIG['variant'],
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_10m': sfi_10m,
                'sfi_160m': sfi_160m,
                'storm_cost': storm_cost,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_10m:+5.2f}  {sfi_160m:+6.2f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        # Log band-dependent SFI at milestone epochs
        if epoch in [1, 10, 25, 50, 75, 100]:
            log.info(f"  SFI by band (SFI=200): {sfi_effects}")

        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    sfi_10m, storm_cost, epoch_sec, is_best)

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    wall_seconds = time.perf_counter() - training_start_time

    audit_complete(run_id, CONFIG, train_size, val_size, date_range,
                   best_rmse, best_pearson, best_sfi_10m, best_kp, best_epoch, wall_seconds)

    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    log.info("")
    log.info("=" * 70)
    log.info("V25-BETA RESULTS")
    log.info("=" * 70)

    log.info("")
    log.info("BAND-DEPENDENT SFI EFFECT (The Key Metric):")
    log.info(f"  10m (high band):  {best_sfi_10m:+.3f}σ")
    log.info(f"  160m (low band):  {best_sfi_160m:+.3f}σ")
    log.info(f"  Delta (10m-160m): {best_sfi_10m - best_sfi_160m:+.3f}σ")

    if abs(best_sfi_10m - best_sfi_160m) > 0.1:
        log.info("  >>> SFI sidecar shows band differentiation! <<<")
    else:
        log.info("  >>> WARNING: SFI effect not differentiated by band <<<")

    log.info("")
    log.info(">>> Run TST-900 to verify physics discrimination <<<")


if __name__ == '__main__':
    train()
