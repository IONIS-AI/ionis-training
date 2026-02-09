#!/usr/bin/env python3
"""
coverage_heatmap.py — Generate global coverage heatmaps from ClickHouse

Creates visualization of WSPR terrestrial coverage vs balloon coverage.
Queries ClickHouse directly for grid coverage data with balloon filtering.

Usage:
  python coverage_heatmap.py                    # Generate both heatmaps
  python coverage_heatmap.py --terrestrial      # Terrestrial only
  python coverage_heatmap.py --balloon          # Balloon only
  python coverage_heatmap.py --output figures/  # Custom output directory

Data sources (ClickHouse):
  - wspr.bronze: Raw WSPR spots (10.8B)
  - wspr.balloon_callsigns: Flagged balloon/telemetry callsigns (1.51M)

Output:
  - terrestrial_coverage.png: Global heatmap of ground station coverage
  - balloon_coverage.png: Global heatmap of balloon/HAP coverage
  - coverage_comparison.png: Side-by-side comparison
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import clickhouse_connect

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(TRAINING_DIR, "figures")

CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

# Grid dimensions
TOTAL_GRIDS = 32400  # 18 * 18 * 10 * 10

# Color maps
TERRESTRIAL_CMAP = "YlOrRd"  # Yellow-Orange-Red for coverage density
BALLOON_CMAP = "Blues"       # Blues for balloon coverage
SILENT_COLOR = "#2d2d2d"     # Dark gray for silent zones


def grid4_to_latlon(g4: str):
    """Convert 4-char Maidenhead grid to (lat, lon) centroid."""
    # Strip null bytes from FixedString columns
    g = g4.replace('\x00', '').strip().upper()
    if len(g) < 4:
        return None, None
    try:
        lon = (ord(g[0]) - ord('A')) * 20.0 - 180.0 + int(g[2]) * 2.0 + 1.0
        lat = (ord(g[1]) - ord('A')) * 10.0 - 90.0 + int(g[3]) * 1.0 + 0.5
        return lat, lon
    except (ValueError, IndexError):
        return None, None


def query_terrestrial_coverage(client) -> pd.DataFrame:
    """Query terrestrial coverage (excluding balloon callsigns)."""
    print("  Querying terrestrial coverage (excluding balloons)...")

    query = """
    SELECT
        substring(grid, 1, 4) AS g4,
        count(*) AS spots
    FROM wspr.bronze
    WHERE callsign NOT IN (SELECT callsign FROM wspr.balloon_callsigns)
      AND length(grid) >= 4
    GROUP BY g4
    ORDER BY spots DESC
    """

    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=['g4', 'spots'])

    # Add lat/lon (filter invalid grids)
    coords = df['g4'].apply(grid4_to_latlon)
    df['lat'] = coords.apply(lambda x: x[0])
    df['lon'] = coords.apply(lambda x: x[1])
    df = df.dropna(subset=['lat', 'lon'])

    return df


def query_balloon_coverage(client) -> pd.DataFrame:
    """Query balloon-only coverage."""
    print("  Querying balloon coverage...")

    query = """
    SELECT
        substring(grid, 1, 4) AS g4,
        count(*) AS spots
    FROM wspr.bronze
    WHERE callsign IN (SELECT callsign FROM wspr.balloon_callsigns)
      AND length(grid) >= 4
    GROUP BY g4
    ORDER BY spots DESC
    """

    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=['g4', 'spots'])

    # Add lat/lon (filter invalid grids)
    coords = df['g4'].apply(grid4_to_latlon)
    df['lat'] = coords.apply(lambda x: x[0])
    df['lon'] = coords.apply(lambda x: x[1])
    df = df.dropna(subset=['lat', 'lon'])

    return df


def create_heatmap(df: pd.DataFrame, title: str, cmap: str, output_path: str,
                   log_scale: bool = True):
    """Create a global heatmap from grid data."""

    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    if log_scale and df['spots'].max() > 0:
        norm = mcolors.LogNorm(vmin=max(1, df['spots'].min()),
                                vmax=df['spots'].max())
    else:
        norm = mcolors.Normalize(vmin=0, vmax=df['spots'].max())

    cmap_obj = plt.get_cmap(cmap)

    # Plot each grid as a rectangle
    for _, row in df.iterrows():
        lon = row['lon'] - 1  # Shift to grid corner
        lat = row['lat'] - 0.5
        spots = row['spots']

        if spots > 0:
            color = cmap_obj(norm(spots))
        else:
            color = SILENT_COLOR

        rect = Rectangle((lon, lat), 2, 1,
                         facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(rect)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')

    # Grid lines
    ax.grid(True, color='#444444', linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))

    # Labels
    ax.set_xlabel('Longitude', color='white', fontsize=12)
    ax.set_ylabel('Latitude', color='white', fontsize=12)
    ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(colors='white')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Spots (log scale)' if log_scale else 'Spots',
                   color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Stats annotation
    covered_grids = len(df[df['spots'] > 0])
    total_spots = df['spots'].sum()
    coverage_pct = 100.0 * covered_grids / TOTAL_GRIDS

    stats_text = (f"Grids: {covered_grids:,} / {TOTAL_GRIDS:,} ({coverage_pct:.1f}%)\n"
                  f"Total spots: {total_spots:,.0f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            color='white', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_path}")
    return covered_grids, total_spots


def create_comparison(terrestrial_df: pd.DataFrame, balloon_df: pd.DataFrame,
                      output_path: str):
    """Create side-by-side comparison of terrestrial vs balloon coverage."""

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='#1a1a1a')

    for ax, df, title, cmap in [
        (axes[0], terrestrial_df, "Terrestrial Coverage (Ground Stations)", TERRESTRIAL_CMAP),
        (axes[1], balloon_df, "Balloon Coverage (High Altitude)", BALLOON_CMAP),
    ]:
        ax.set_facecolor('#1a1a1a')

        if len(df) > 0 and df['spots'].max() > 0:
            norm = mcolors.LogNorm(vmin=max(1, df[df['spots'] > 0]['spots'].min()),
                                    vmax=df['spots'].max())
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

        cmap_obj = plt.get_cmap(cmap)

        for _, row in df.iterrows():
            lon = row['lon'] - 1
            lat = row['lat'] - 0.5
            spots = row['spots']

            if spots > 0:
                color = cmap_obj(norm(spots))
            else:
                color = SILENT_COLOR

            rect = Rectangle((lon, lat), 2, 1,
                             facecolor=color, edgecolor='none', alpha=0.9)
            ax.add_patch(rect)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect('equal')
        ax.grid(True, color='#444444', linewidth=0.5, alpha=0.5)
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')

        # Stats
        covered = len(df[df['spots'] > 0])
        pct = 100.0 * covered / TOTAL_GRIDS
        ax.text(0.02, 0.98, f"{covered:,} grids ({pct:.1f}%)",
                transform=ax.transAxes, color='white', fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    plt.suptitle("IONIS Coverage Analysis — Terrestrial vs Balloon",
                 color='white', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', edgecolor='none',
                bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate WSPR coverage heatmaps from ClickHouse")
    parser.add_argument("--terrestrial", action="store_true",
                        help="Generate terrestrial heatmap only")
    parser.add_argument("--balloon", action="store_true",
                        help="Generate balloon heatmap only")
    parser.add_argument("--output", default=FIGURES_DIR,
                        help="Output directory for figures")
    parser.add_argument("--host", default=CH_HOST,
                        help="ClickHouse host")
    parser.add_argument("--port", type=int, default=CH_PORT,
                        help="ClickHouse port")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  IONIS Coverage Heatmap Generator")
    print("=" * 70)

    # Connect to ClickHouse
    print(f"\nConnecting to ClickHouse at {args.host}:{args.port}...")
    client = clickhouse_connect.get_client(host=args.host, port=args.port)

    # Determine what to generate
    do_terrestrial = args.terrestrial or (not args.terrestrial and not args.balloon)
    do_balloon = args.balloon or (not args.terrestrial and not args.balloon)
    do_comparison = do_terrestrial and do_balloon

    terrestrial_df = None
    balloon_df = None

    if do_terrestrial:
        print(f"\nLoading terrestrial coverage...")
        terrestrial_df = query_terrestrial_coverage(client)
        print(f"  Loaded {len(terrestrial_df):,} grid records")

        output_path = os.path.join(args.output, "terrestrial_coverage.png")
        print(f"\nGenerating terrestrial heatmap...")
        grids, spots = create_heatmap(
            terrestrial_df,
            "IONIS Terrestrial Coverage — Ground Station WSPR (Excluding Balloons)",
            TERRESTRIAL_CMAP,
            output_path
        )
        print(f"  Coverage: {grids:,} grids ({100*grids/TOTAL_GRIDS:.1f}%), {spots:,.0f} spots")

    if do_balloon:
        print(f"\nLoading balloon coverage...")
        balloon_df = query_balloon_coverage(client)
        print(f"  Loaded {len(balloon_df):,} grid records")

        output_path = os.path.join(args.output, "balloon_coverage.png")
        print(f"\nGenerating balloon heatmap...")
        grids, spots = create_heatmap(
            balloon_df,
            "IONIS Balloon Coverage — High Altitude WSPR",
            BALLOON_CMAP,
            output_path
        )
        print(f"  Coverage: {grids:,} grids ({100*grids/TOTAL_GRIDS:.1f}%), {spots:,.0f} spots")

    if do_comparison and terrestrial_df is not None and balloon_df is not None:
        print(f"\nGenerating comparison figure...")
        output_path = os.path.join(args.output, "coverage_comparison.png")
        create_comparison(terrestrial_df, balloon_df, output_path)

    print("\n" + "=" * 70)
    print("  Heatmap generation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
