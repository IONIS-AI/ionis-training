#!/bin/bash
# IONIS V20 Link Budget Validation Battery
# 24 profiles x 3 sources = 72 scoring runs
#
# Usage: bash tools/run_battery.sh 2>&1 | tee battery_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

CONFIG="versions/v20/config_v20.json"
PYTHON="${PYTHON:-python}"
SCORER="tools/score_model.py"

PROFILES=(
    wspr wspr_dipole voacap_default
    qrp_milliwatt qrp_portable qrp_home sota_activator pota_activator
    home_vertical home_station home_beam
    home_amp_dipole home_amp_beam big_gun
    contest_lp contest_cw contest_ssb contest_super
    dxpedition_lite dxpedition dxpedition_mega
    maritime_mobile eme_hf
)

SOURCES=(rbn pskr contest)

echo "======================================================================"
echo "  IONIS V20 Link Budget Validation Battery"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Profiles: ${#PROFILES[@]}  Sources: ${#SOURCES[@]}"
echo "  Total runs: $(( ${#PROFILES[@]} * ${#SOURCES[@]} ))"
echo "======================================================================"

PASS=0
FAIL=0
RUN=0

for source in "${SOURCES[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Source: ${source}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for profile in "${PROFILES[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "── Run ${RUN}/$(( ${#PROFILES[@]} * ${#SOURCES[@]} )): ${source} x ${profile} ──"
        if $PYTHON "$SCORER" --config "$CONFIG" --source "$source" --profile "$profile"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            echo "  *** FAILED: ${source} x ${profile} ***"
        fi
    done
done

echo ""
echo "======================================================================"
echo "  BATTERY COMPLETE — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Total: ${RUN}  Pass: ${PASS}  Fail: ${FAIL}"
echo "======================================================================"
