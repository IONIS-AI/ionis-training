-- ==========================================================================
-- pskr_failure_analysis.sql — PSKR Validation Diagnostic Queries
-- ==========================================================================
-- Run against validation.model_results after scoring PSKR signatures
-- with score_model.py. Each query identifies a different failure dimension.
--
-- Prerequisites:
--   1. Score with override:
--      python tools/score_model.py --config versions/v22/config_v22.json \
--          --source pskr_sig --profile wspr
--
--   2. Score without override (for Query 1):
--      python tools/score_model.py --config versions/v22/config_v22.json \
--          --source pskr_sig --profile wspr --no-override
--
-- Seasonal limitation: Oct 2025 – Feb 2026 only (solar max winter).
-- No summer data — results should not be extrapolated to July.
-- ==========================================================================


-- ── Query 1: Override Impact (THE MONEY QUERY) ─────────────────────────────
-- Paths the PhysicsOverrideLayer closed that PSKR proves were actually open.
-- Requires both pskr_sig and pskr_sig_no_override scoring runs.
--
-- override_caused_miss = override closed a path that was open (false negative)
-- override_fixed_miss  = override fixed a path the model got wrong (true negative)
-- ==========================================================================

SELECT
    a.band,
    a.hour,
    count()                                              AS total,
    countIf(a.mode_hit = 0 AND b.mode_hit = 1)          AS override_caused_miss,
    countIf(a.mode_hit = 1 AND b.mode_hit = 0)          AS override_fixed_miss,
    round(100.0 * countIf(a.mode_hit = 0 AND b.mode_hit = 1) / count(), 2)
                                                         AS override_miss_pct
FROM validation.model_results a
JOIN validation.model_results b
    ON  a.tx_grid_4 = b.tx_grid_4
    AND a.rx_grid_4 = b.rx_grid_4
    AND a.band      = b.band
    AND a.hour      = b.hour
    AND a.month     = b.month
WHERE a.source = 'pskr_sig'              AND a.model_version = 'v22'
  AND b.source = 'pskr_sig_no_override'  AND b.model_version = 'v22'
GROUP BY a.band, a.hour
HAVING override_caused_miss > 0
ORDER BY override_caused_miss DESC
LIMIT 50
FORMAT PrettyCompact;


-- ── Query 2: False Negative Rate by Band × Hour ────────────────────────────
-- "10m at 08z has 45% miss rate" = model thinks band is closed when
-- PSKR proves it's open.
-- ==========================================================================

SELECT
    band,
    hour,
    count()                                AS total,
    countIf(mode_hit = 0)                  AS misses,
    round(100.0 * countIf(mode_hit = 0) / count(), 2) AS miss_pct,
    round(avg(snr_error), 2)               AS avg_bias
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
GROUP BY band, hour
ORDER BY miss_pct DESC
LIMIT 50
FORMAT PrettyCompact;


-- ── Query 3: Worst Failures by Distance Bracket ────────────────────────────
-- Long-path propagation gaps, NVIS misses, distance-dependent bias.
-- ==========================================================================

SELECT
    band,
    multiIf(
        distance_km < 2000, 'short (<2k)',
        distance_km < 8000, 'medium (2-8k)',
        'long (>8k)'
    ) AS dist_class,
    count()                                AS total,
    round(100.0 * countIf(mode_hit = 0) / count(), 2) AS miss_pct,
    round(avg(snr_error), 2)               AS avg_bias
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
GROUP BY band, dist_class
ORDER BY miss_pct DESC
FORMAT PrettyCompact;


-- ── Query 4: Failure Clusters by SFI Bracket ───────────────────────────────
-- Does the scalar SFI sidecar (+0.48σ) underpredict at high SFI on
-- specific bands?
-- ==========================================================================

SELECT
    band,
    multiIf(
        avg_sfi < 80,  'low (<80)',
        avg_sfi < 120, 'moderate (80-120)',
        avg_sfi < 160, 'elevated (120-160)',
        'high (>160)'
    ) AS sfi_class,
    count()                                AS total,
    round(100.0 * countIf(mode_hit = 0) / count(), 2) AS miss_pct,
    round(avg(snr_error), 2)               AS avg_bias
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
GROUP BY band, sfi_class
ORDER BY miss_pct DESC
FORMAT PrettyCompact;


-- ── Query 5: Geographic Failure Heatmap (TX Grid Field) ─────────────────────
-- Model may underpredict from equatorial regions (high foF2) or
-- overpredict from high latitudes (auroral absorption).
-- ==========================================================================

SELECT
    substring(tx_grid_4, 1, 2) AS tx_field,
    band,
    count()                                AS total,
    round(100.0 * countIf(mode_hit = 0) / count(), 2) AS miss_pct,
    round(avg(snr_error), 2)               AS avg_bias
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
GROUP BY tx_field, band
HAVING total >= 100
ORDER BY miss_pct DESC
LIMIT 30
FORMAT PrettyCompact;


-- ── Query 6: Per-Mode Recall Comparison ─────────────────────────────────────
-- FT8 vs CW vs WSPR vs RTTY recall by band.
-- ==========================================================================

SELECT
    actual_mode,
    band,
    count()                                AS total,
    round(100.0 * countIf(mode_hit = 1) / count(), 2) AS recall_pct,
    round(avg(predicted_snr), 2)           AS avg_pred,
    round(avg(actual_snr), 2)              AS avg_actual
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
GROUP BY actual_mode, band
ORDER BY actual_mode, band
FORMAT PrettyCompact;


-- ── Query 7: Seasonal Failure Patterns by Month ─────────────────────────────
-- Does accuracy change with season? Oct–Feb spans equinox→solstice→post.
-- Different D-layer profiles, day lengths, and greyline windows.
-- ==========================================================================

SELECT
    band,
    month,
    count()                                AS total,
    countIf(mode_hit = 0)                  AS misses,
    round(100.0 * countIf(mode_hit = 0) / count(), 2) AS miss_pct,
    round(avg(snr_error), 2)               AS avg_bias
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
GROUP BY band, month
ORDER BY band, month
FORMAT PrettyCompact;


-- ── Query 8: Overall Summary (Quick Health Check) ───────────────────────────
-- One-row summary: total scored, overall recall, RMSE, bias.
-- ==========================================================================

SELECT
    count()                                              AS total_scored,
    countIf(mode_hit = 1)                                AS mode_hits,
    round(100.0 * countIf(mode_hit = 1) / count(), 2)   AS recall_pct,
    round(sqrt(avg(snr_error * snr_error)), 2)           AS rmse_db,
    round(avg(snr_error), 2)                             AS bias_db,
    round(avg(predicted_snr), 2)                         AS avg_pred_snr,
    round(avg(actual_snr), 2)                            AS avg_actual_snr
FROM validation.model_results
WHERE model_version = 'v22' AND source = 'pskr_sig'
FORMAT PrettyCompact;
