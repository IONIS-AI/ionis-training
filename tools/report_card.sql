-- ==============================================================================
-- report_card.sql — IONIS Model Validation Report Card
--
-- Queries against validation.model_results to produce the three-tier
-- report: Strengths (green, >= 90%) / Adequate (yellow, 70-89%) / Weak (red, < 70%)
--
-- Usage:
--   clickhouse-client --multiquery < tools/report_card.sql
--   clickhouse-client -q "..." for individual queries
--
-- Requires: validation.model_results populated by tools/score_model.py
--
-- NOTE on snr_error:
--   - PSKR: predicted (WSPR-scale) vs observed (machine-decoded) — comparable
--   - RBN:  predicted (WSPR-scale) vs observed (RBN-scale) — includes ~35 dB offset
--   - Contest: predicted vs threshold (surrogate) — mode_hit is the key metric
-- ==============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. SCORING RUN INVENTORY
-- ─────────────────────────────────────────────────────────────────────────────
-- What runs exist? Sanity check before analysis.

SELECT
    model_version,
    source,
    run_id,
    run_timestamp,
    count()                                    AS total_paths,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct
FROM validation.model_results
GROUP BY model_version, source, run_id, run_timestamp
ORDER BY model_version, source, run_timestamp DESC
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. OVERALL RECALL BY MODE
-- ─────────────────────────────────────────────────────────────────────────────
-- Core metric: for each mode observed, what fraction did the model predict
-- as viable? This is the primary report card.

SELECT
    actual_mode,
    count()                                    AS total,
    sum(mode_hit)                              AS hits,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 'STRENGTH',
        sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
        'WEAK'
    )                                          AS tier
FROM validation.model_results
WHERE model_version = 'v20'
GROUP BY actual_mode
ORDER BY recall_pct DESC
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. PER-BAND RECALL BY SOURCE
-- ─────────────────────────────────────────────────────────────────────────────
-- The geographic/propagation view. Where is the model strong and weak?

SELECT
    multiIf(
        band = 102, '160m', band = 103, '80m', band = 104, '60m',
        band = 105, '40m',  band = 106, '30m', band = 107, '20m',
        band = 108, '17m',  band = 109, '15m', band = 110, '12m',
        band = 111, '10m',  toString(band)
    )                                          AS band_name,
    source,
    count()                                    AS total,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    round(avg(snr_error), 2)                   AS mean_bias_db,
    round(sqrt(avg(snr_error * snr_error)), 2) AS rmse_db,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 'STRENGTH',
        sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
        'WEAK'
    )                                          AS tier
FROM validation.model_results
WHERE model_version = 'v20'
GROUP BY band, source
ORDER BY band, source
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. SOLAR CONDITION BREAKDOWN (STORM PERFORMANCE)
-- ─────────────────────────────────────────────────────────────────────────────
-- How does the model perform under different geomagnetic conditions?
-- Kp < 2 = quiet, 2-5 = unsettled, >= 5 = storm

SELECT
    multiIf(avg_kp < 2, 'quiet', avg_kp < 5, 'unsettled', 'storm') AS kp_class,
    source,
    count()                                    AS total,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    round(avg(snr_error), 2)                   AS mean_bias_db,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 'STRENGTH',
        sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
        'WEAK'
    )                                          AS tier
FROM validation.model_results
WHERE model_version = 'v20'
GROUP BY kp_class, source
ORDER BY kp_class, source
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. SFI CONDITION BREAKDOWN
-- ─────────────────────────────────────────────────────────────────────────────
-- Model behavior across the solar cycle: low/mid/high SFI

SELECT
    multiIf(avg_sfi < 100, 'low (<100)', avg_sfi < 150, 'mid (100-150)', 'high (>150)') AS sfi_class,
    source,
    count()                                    AS total,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    round(avg(snr_error), 2)                   AS mean_bias_db,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 'STRENGTH',
        sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
        'WEAK'
    )                                          AS tier
FROM validation.model_results
WHERE model_version = 'v20'
GROUP BY sfi_class, source
ORDER BY sfi_class, source
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. DISTANCE-BAND HEATMAP (THE GAP FINDER)
-- ─────────────────────────────────────────────────────────────────────────────
-- 2000 km distance buckets x band. Finds the specific propagation
-- conditions where the model struggles.

SELECT
    multiIf(
        band = 102, '160m', band = 103, '80m', band = 104, '60m',
        band = 105, '40m',  band = 106, '30m', band = 107, '20m',
        band = 108, '17m',  band = 109, '15m', band = 110, '12m',
        band = 111, '10m',  toString(band)
    )                                          AS band_name,
    intDiv(distance_km, 2000) * 2000           AS dist_bucket_km,
    count()                                    AS total,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 'STRENGTH',
        sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
        'WEAK'
    )                                          AS tier
FROM validation.model_results
WHERE model_version = 'v20'
  AND source = 'rbn'
GROUP BY band, dist_bucket_km
HAVING total >= 100
ORDER BY band, dist_bucket_km
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. TIME-OF-DAY PERFORMANCE
-- ─────────────────────────────────────────────────────────────────────────────
-- Day vs night recall. The model's day_night_est feature should capture this.

SELECT
    multiIf(hour >= 6 AND hour < 18, 'day (06-18)', 'night (18-06)') AS time_class,
    source,
    count()                                    AS total,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    round(avg(snr_error), 2)                   AS mean_bias_db
FROM validation.model_results
WHERE model_version = 'v20'
GROUP BY time_class, source
ORDER BY time_class, source
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8. VIABILITY WATERFALL CONSISTENCY CHECK
-- ─────────────────────────────────────────────────────────────────────────────
-- SSB viable implies CW viable implies FT8 viable.
-- Any violation means a bug in the scoring pipeline.

SELECT
    'ssb_viable=1 AND cw_viable=0'  AS violation,
    count()                         AS count
FROM validation.model_results
WHERE model_version = 'v20'
  AND ssb_viable = 1 AND cw_viable = 0

UNION ALL

SELECT
    'cw_viable=1 AND ft8_viable=0'  AS violation,
    count()                         AS count
FROM validation.model_results
WHERE model_version = 'v20'
  AND cw_viable = 1 AND ft8_viable = 0

UNION ALL

SELECT
    'rtty_viable=1 AND ft8_viable=0' AS violation,
    count()                          AS count
FROM validation.model_results
WHERE model_version = 'v20'
  AND rtty_viable = 1 AND ft8_viable = 0

FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 9. THREE-TIER SUMMARY (THE FINAL REPORT CARD)
-- ─────────────────────────────────────────────────────────────────────────────
-- Aggregates everything into a single view: band x source x tier.
-- Filters to cells with >= 1000 samples for statistical significance.

SELECT
    multiIf(
        band = 102, '160m', band = 103, '80m', band = 104, '60m',
        band = 105, '40m',  band = 106, '30m', band = 107, '20m',
        band = 108, '17m',  band = 109, '15m', band = 110, '12m',
        band = 111, '10m',  toString(band)
    )                                          AS band_name,
    source,
    count()                                    AS total,
    round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
    round(avg(predicted_snr), 1)               AS avg_pred_snr,
    round(avg(actual_snr), 1)                  AS avg_actual_snr,
    round(sqrt(avg(snr_error * snr_error)), 2) AS rmse_db,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 'STRENGTH',
        sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
        'WEAK'
    )                                          AS tier
FROM validation.model_results
WHERE model_version = 'v20'
GROUP BY band, source
HAVING total >= 1000
ORDER BY
    source,
    multiIf(
        sum(mode_hit) / count() >= 0.90, 1,
        sum(mode_hit) / count() >= 0.70, 2,
        3
    ),
    band
FORMAT PrettyCompact;


-- ─────────────────────────────────────────────────────────────────────────────
-- 10. TIER DISTRIBUTION SUMMARY
-- ─────────────────────────────────────────────────────────────────────────────
-- How many band x source cells fall in each tier?

SELECT
    tier,
    count()                  AS cells,
    sum(total)               AS total_paths,
    round(avg(recall_pct), 2) AS avg_recall_pct
FROM (
    SELECT
        band,
        source,
        count()                                    AS total,
        round(sum(mode_hit) / count() * 100, 2)   AS recall_pct,
        multiIf(
            sum(mode_hit) / count() >= 0.90, 'STRENGTH',
            sum(mode_hit) / count() >= 0.70, 'ADEQUATE',
            'WEAK'
        )                                          AS tier
    FROM validation.model_results
    WHERE model_version = 'v20'
    GROUP BY band, source
    HAVING total >= 1000
)
GROUP BY tier
ORDER BY tier
FORMAT PrettyCompact;
