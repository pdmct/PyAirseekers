#!/usr/bin/env python3
"""
Overlay Airseekers Tron mower map data on an orthophoto.
Uses windowed reading for large ortho TIFFs.
"""

import asyncio
import json
import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiohttp
from api_client import AirseekersAPI

SN = os.environ.get('AIRSEEKERS_SN', '')
if not SN:
    raise ValueError("Set AIRSEEKERS_SN environment variable to your device serial number")
ORTHO_CRS = "EPSG:32755"

# Feature type mapping
TYPE_NAMES = {
    1: 'mowing_zone',
    2: 'exclusion_zone',
    3: 'path',
    4: 'dock_exclusion',
    5: 'dock_pad',
    6: 'charge_point',
    7: 'undock_point',
    8: 'reference_point',
    9: 'obstacle',
}

COLORS = {
    'mowing_zone': {'face': 'lime', 'edge': 'lime', 'alpha': 0.2},
    'exclusion_zone': {'face': 'red', 'edge': 'red', 'alpha': 0.3},
    'path': {'color': 'yellow', 'linewidth': 2.5, 'linestyle': '--'},
    'dock_pad': {'face': 'cyan', 'edge': 'blue', 'alpha': 0.5},
    'dock_exclusion': {'face': 'none', 'edge': 'cyan', 'alpha': 0.3, 'linestyle': ':'},
    'obstacle': {'face': 'orange', 'edge': 'darkorange', 'alpha': 0.4},
}


def local_to_utm(x_local, y_local, dock_e, dock_n, heading_rad):
    """Convert mower local coords to UTM.

    The local frame has the dock at origin.
    heading_rad measures dock's forward direction (clockwise from north).
    x_local is east-west offset (east is +), y_local is north-south offset.
    """

    x_offset, y_offset = (1.57, -0.56)  # Align dock with mower
    x_local += x_offset
    y_local += y_offset

    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    
    easting = dock_e + x_local * cos_h - y_local * sin_h
    northing = dock_n + x_local * sin_h + y_local * cos_h

    return easting, northing


async def fetch_data():
    async with aiohttp.ClientSession() as session:
        api = AirseekersAPI(
            os.environ['AIRSEEKERS_EMAIL'],
            os.environ['AIRSEEKERS_PASSWORD'],
            session=session
        )
        await api.get_server_host()
        await api.login()

        map_resp = await api._make_request("GET", f"/api/web/device/map?sn={SN}")
        status_resp = await api._make_request("GET", f"/api/web/device/full-status?sn={SN}")
        return map_resp.data, status_resp.data.get('rtk_status', {})


def extract_xy(coords):
    """Extract x,y values from coordinates that may have extra elements."""
    points = []
    for c in coords:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            points.append((c[0], c[1]))
    return points


def render_overlay(ortho_path, output_path, heading_override=None, padding=15, rot_offset=0.0, debug=False, flip_x=False, flip_y=False, dx=0.0, dy=0.0):
    print("📡 Fetching mower data...")
    map_data, rtk = asyncio.run(fetch_data())

    # Use active map (Map-1)
    active_map = map_data[0]
    features = active_map['geoData']['features']
    print(f"🗺️  Map: {active_map['mapName']}, {len(features)} features")

    # Look for dock location
    dock_lat, dock_lon, dock_heading = None, None, None
    for feat in features:
        if feat["properties"]["type"] == 6:
            dock_lat, dock_lon, dock_heading = feat["geometry"]["coordinates"][3], feat["geometry"]["coordinates"][2], feat["geometry"]["coordinates"][4]
            print(f"  Located Dock Lat={dock_lat}, Lon={dock_lon}, Heading={math.degrees(dock_heading):.2f}°")
            break
    if not dock_lat:
        raise ValueError("💔 No Charge Point")

    if heading_override is not None:
        dock_heading = math.radians(heading_override)
        print(f"  Heading override: {heading_override}°")

    dock_heading += math.radians(rot_offset)
    if rot_offset != 0:
        print(f"  Rotation offset: {rot_offset}° → effective heading: {math.degrees(dock_heading):.2f}°")

    # Convert dock GPS to UTM
    # always_xy=True means we pass (lon, lat) and get (easting, northing)
    transformer = Transformer.from_crs("EPSG:4326", ORTHO_CRS, always_xy=True)
    dock_e, dock_n = transformer.transform(dock_lon, dock_lat)
    dock_e += dx
    dock_n += dy
    print(f"  Dock UTM: E={dock_e:.2f}, N={dock_n:.2f} (nudge: dx={dx:+.2f}, dy={dy:+.2f})")

    if debug:
        print(f"\n  [DEBUG] Raw charge_point: lat={dock_lat}, lon={dock_lon}, heading_rad={dock_heading:.4f} ({math.degrees(dock_heading):.2f}°)")
        print(f"  [DEBUG] Dock UTM (EPSG:32755): E={dock_e:.3f}, N={dock_n:.3f}\n")

    # Convert all features to UTM
    utm_features = []
    all_e, all_n = [dock_e], [dock_n]

    for feat in features:
        ftype = feat["properties"]["type"]
        fname = TYPE_NAMES.get(ftype, f'type_{ftype}')
        coords = feat["geometry"]["coordinates"]

        if ftype == 6:
            # Charge point — single point at dock
            utm_features.append({'name': fname, 'type': ftype, 'kind': 'point', 'e': dock_e, 'n': dock_n})
            continue

        if ftype in (7, 8):
            # Single point features in local coords
            if isinstance(coords[0], (int, float)):
                e, n = local_to_utm(coords[0], coords[1], dock_e, dock_n, dock_heading)
                n -= 0.5  # nudge 0.5m south
                utm_features.append({'name': fname, 'type': ftype, 'kind': 'point', 'e': e, 'n': n})
                all_e.append(e); all_n.append(n)
            continue

        # Polygon/polyline features
        # GeoJSON polygons are [[[x,y],...]] (list of rings) — unwrap outer ring
        raw = coords[0] if coords and isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (list, tuple)) else coords
        pts = extract_xy(raw)
        if not pts:
            continue
        fx = -1 if flip_x else 1
        fy = -1 if flip_y else 1
        utm_pts = [local_to_utm(fx * x, fy * y, dock_e, dock_n, dock_heading) for x, y in pts]
        es = [p[0] for p in utm_pts]
        ns = [p[1] for p in utm_pts]
        all_e.extend(es); all_n.extend(ns)
        utm_features.append({'name': fname, 'type': ftype, 'kind': 'polygon' if ftype != 3 else 'line', 'es': es, 'ns': ns})

    # Bounding box with padding
    min_e, max_e = min(all_e) - padding, max(all_e) + padding
    min_n, max_n = min(all_n) - padding, max(all_n) + padding
    print(f"  Bounds (UTM): E=[{min_e:.1f}, {max_e:.1f}], N=[{min_n:.1f}, {max_n:.1f}]")

    # Read ortho window
    print(f"🖼️  Reading ortho: {ortho_path}")
    with rasterio.open(ortho_path) as src:
        window = from_bounds(min_e, min_n, max_e, max_n, src.transform)
        data = src.read([1, 2, 3], window=window)
        win_transform = src.window_transform(window)

    # data shape: (3, H, W) — convert to (H, W, 3) for imshow
    img = np.moveaxis(data, 0, -1)
    # Normalise to 0-1 if needed
    if img.dtype != np.uint8:
        img = np.clip(img / img.max(), 0, 1)

    H, W = img.shape[:2]

    # Get pixel-aligned actual bounds from the window transform
    # Rasterio stores rows top→bottom (row 0 = max northing)
    win_left  = win_transform.c
    win_top   = win_transform.f
    win_right = win_left + W * win_transform.a
    win_bot   = win_top  + H * win_transform.e   # e is negative → south edge

    if debug:
        print(f"  [DEBUG] Window extent: E=[{win_left:.2f}, {win_right:.2f}], N=[{win_bot:.2f}, {win_top:.2f}]")

    dpi = 150
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    # origin='upper' matches rasterio row order (row 0 at top = max northing)
    ax.imshow(img, extent=[win_left, win_right, win_bot, win_top], origin='upper', aspect='equal')
    ax.set_xlim(win_left, win_right)
    ax.set_ylim(win_bot, win_top)
    ax.axis('off')

    # Draw features
    legend_handles = []
    drawn_names = set()

    for feat in utm_features:
        name = feat['name']
        ftype = feat['type']
        style = COLORS.get(name, {})

        if feat['kind'] == 'polygon':
            poly = plt.Polygon(list(zip(feat['es'], feat['ns'])),
                               facecolor=style.get('face', 'white'),
                               edgecolor=style.get('edge', 'white'),
                               alpha=style.get('alpha', 0.3),
                               linestyle=style.get('linestyle', '-'),
                               linewidth=1.5, closed=True)
            ax.add_patch(poly)
            if name not in drawn_names:
                legend_handles.append(mpatches.Patch(facecolor=style.get('face', 'white'),
                                                      edgecolor=style.get('edge', 'white'),
                                                      alpha=min(style.get('alpha', 0.3) + 0.3, 1.0),
                                                      label=name.replace('_', ' ').title()))
                drawn_names.add(name)

        elif feat['kind'] == 'line':
            ax.plot(feat['es'], feat['ns'],
                    color=style.get('color', 'yellow'),
                    linewidth=style.get('linewidth', 2),
                    linestyle=style.get('linestyle', '--'))
            if name not in drawn_names:
                legend_handles.append(plt.Line2D([0], [0],
                                                  color=style.get('color', 'yellow'),
                                                  linewidth=style.get('linewidth', 2),
                                                  linestyle=style.get('linestyle', '--'),
                                                  label=name.replace('_', ' ').title()))
                drawn_names.add(name)

        elif feat['kind'] == 'point':
            # Charge point (type 6): nudge marker to north side of dock pad
            if ftype == 6:
                pe, pn = dock_e - 1.5, dock_n  # 1.5m west of dock origin
            else:
                pe, pn = feat['e'], feat['n']
            color = 'cyan' if ftype == 6 else ('white' if ftype == 7 else 'magenta')
            marker = 'D' if ftype == 6 else 'o'
            ax.plot(pe, pn, marker=marker, color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=10)
            if name not in drawn_names:
                legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w',
                                                  markerfacecolor=color, markeredgecolor='black',
                                                  markersize=8, label=name.replace('_', ' ').title()))
                drawn_names.add(name)

    # Plot mower position using local pose coords (consistent with map frame)
    if rtk and rtk.get('robot_pose_x') is not None and rtk.get('robot_pose_y') is not None:
        fx = -1 if flip_x else 1
        fy = -1 if flip_y else 1
        mow_e, mow_n = local_to_utm(fx * rtk['robot_pose_x'], fy * rtk['robot_pose_y'], dock_e, dock_n, dock_heading)
        mow_n -= 0.5  # nudge 0.5m south
        ax.plot(mow_e, mow_n, marker='^', color='yellow', markersize=14,
                markeredgecolor='black', markeredgewidth=1.5, zorder=11)
        legend_handles.append(plt.Line2D([0], [0], marker='^', color='w',
                                          markerfacecolor='yellow', markeredgecolor='black',
                                          markersize=10, label='Mower Position'))
        print(f"  Mower local pose: x={rtk['robot_pose_x']:.3f}, y={rtk['robot_pose_y']:.3f} → E={mow_e:.2f}, N={mow_n:.2f}")
        if debug:
            print(f"  [DEBUG] Dock→Mower delta: dE={mow_e-dock_e:.2f}m, dN={mow_n-dock_n:.2f}m")

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
                  framealpha=0.7, facecolor='black', labelcolor='white')

    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Overlay Airseekers Tron map on orthophoto')
    parser.add_argument('ortho', help='Path to orthophoto TIFF')
    parser.add_argument('output', nargs='?', default='tron_overlay.png', help='Output PNG path (default: tron_overlay.png)')
    parser.add_argument('--heading', type=float, default=None, help='Override dock heading in degrees')
    parser.add_argument('--padding', type=float, default=15, help='Padding around map bounds in metres (default: 15)')
    parser.add_argument('--rot-offset', type=float, default=0.0, dest='rot_offset',
                        help='Add extra rotation offset in degrees to tune alignment')
    parser.add_argument('--flip-x', action='store_true', dest='flip_x', help='Negate local X axis')
    parser.add_argument('--flip-y', action='store_true', dest='flip_y', help='Negate local Y axis')
    parser.add_argument('--dx', type=float, default=0.0, help='Nudge overlay east/west in metres (+east, -west)')
    parser.add_argument('--dy', type=float, default=0.0, help='Nudge overlay north/south in metres (+north, -south)')
    parser.add_argument('--debug', action='store_true', help='Print raw GPS/UTM values for alignment debugging')
    args = parser.parse_args()

    render_overlay(args.ortho, args.output, heading_override=args.heading, padding=args.padding,
                   rot_offset=args.rot_offset, debug=args.debug, flip_x=args.flip_x, flip_y=args.flip_y,
                   dx=args.dx, dy=args.dy)

