#!/usr/bin/env python3
"""
Plots for alasight output. Four modes, one shared set of machinery.

  area      Every region shaded, with its clade count written on it. Reads a
            *_regions_summary.tsv, or any TSV with Q name / Q start / Q end.
            Applies its own size and cluster filters and prints detection
            scores against the known plasmid region.

  depth     Clade depth as a bar track: bar height log2(Depth), so a locus with
            eight clades stacked on it stands a step above one where two
            brushed past each other.

  fraction  Clade breadth: how FEW of the clades that could have been there
            actually were, as -log2(Tree Leaves Hit / Tree Leaves In LCA). A
            gene inherited vertically turns up in most of its clade so the
            fraction is near 1 and the bar is flat; a gene that arrived by
            transfer turns up in a scattered few of a large clade, so the
            fraction is small and the bar is tall.

  mirror    Both at once: depth above the axis, breadth below it. Both halves
            are log2, so one unit is one doubling either way and the two are
            directly comparable. Expect the lower half to run deeper - a
            fraction of 1/500 is nine doublings, while a depth of 8 is three.

Usage:
  python3 alasight_plot.py depth    run_dust_regions_depth.tsv out.svg
  python3 alasight_plot.py mirror   run_dust_regions_depth.tsv out --min-depth 3
  python3 alasight_plot.py fraction run_dust_regions_depth.tsv out --max-fraction 0.01
  python3 alasight_plot.py area     run_dust_regions_summary.tsv out --cluster 500

Output is SVG unless the filename says otherwise; see resolve_output.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize, to_hex
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
except ModuleNotFoundError:
    print("Error: Matplotlib is not installed.")
    print("Please install Matplotlib by running: pip install matplotlib")
    sys.exit(1)


# ===========================================================================
# Constants
# ===========================================================================

# x-axis range when neither the TSV nor a FASTA gives the sequence length.
X_MAX_DEFAULT = 100000

# dpi for raster output. min_draw_width reads it, so the one-pixel floor halves
# when the dpi doubles: a small region is drawn more faithfully at 300 than at
# 150. Vector output ignores it. --dpi overrides.
SAVE_DPI = 300

# Set from the output filename in main. A raster needs the one-pixel floor in
# min_draw_width or sub-pixel regions vanish, but a vector file has no pixel
# grid to accommodate: baking the floor into its geometry would widen every
# small region permanently, and zooming in would show a rectangle several times
# the size of the region it stands for.
VECTOR_OUTPUT = False
VECTOR_EXTS = ('.svg', '.svgz', '.pdf', '.eps', '.ps')
DEFAULT_EXT = '.svg'

# Height of the strip marking a called region, in axis units. It shows where a
# region is when its bar has no height - depth 1, or a fraction of 1.
BASELINE_H = 0.18

# Bars are grouped for drawing by the colour the SVG will actually carry - an
# 8-bit "#rrggbb" - rather than by the value behind it. That is lossless, since
# two values whose colours agree to that precision render identically, and it
# collapses far more than grouping by value does: a depth range of 2..15,467
# is 15,466 distinct values but only 194 distinct colours.

DEPTH_CMAP = 'YlOrRd'          # clade depth: pale = few, dark = many
FRACTION_CMAP = 'YlGnBu'       # clade breadth: pale = most of the clade hit,
                               # dark = a small scattered fraction of it


# ===========================================================================
# Output plumbing
# ===========================================================================

def resolve_output(path):
    """
    (filename, is_vector), adding DEFAULT_EXT when no extension was given.

    SVG by default. Vector output stores every region at its true width, so a
    200 bp region measures 200 bp however far out the view is and the figure
    can be zoomed and measured rather than only glanced at. The cost is
    accepted deliberately: on a whole chromosome a small region is a sub-pixel
    sliver and looks faint until zoomed. A raster cannot do that - it has to
    widen such a region to a whole pixel to show it at all - so .png still gets
    the floor.
    """
    stem, ext = os.path.splitext(path)
    if not ext:
        ext = DEFAULT_EXT
        path = stem + ext
    return path, ext.lower() in VECTOR_EXTS


def min_draw_width(ax, x_max):
    """
    Width in data units of one pixel of this axis, or 0 for vector output.

    A rectangle narrower than a pixel cannot be drawn honestly in a raster. On
    a 5 Mb chromosome an axis holds roughly 1,500 bp per pixel at 300 dpi, so a
    200 bp region is a fraction of a pixel: drawn plainly it antialiases away to
    nothing, and drawn with a 1 pt edge it would paint about 4 px - a 20x
    overstatement, which is what turns a genome with a few small regions into a
    solid band. Widening it to exactly one pixel keeps it visible while
    overstating it as little as a raster allows.
    """
    if VECTOR_OUTPUT:
        return 0.0
    fig = ax.get_figure()
    axis_px = fig.get_size_inches()[0] * SAVE_DPI * ax.get_position().width
    return x_max / max(1.0, axis_px)


def raster_dpi(fig, dpi, warn_mpx=60):
    """
    The dpi to actually use, given how large the raster would be.

    Panels stack, so a hundred-query run is a very tall figure: at 300 dpi that
    is hundreds of megabytes held in memory while it is written. Merely large
    gets a warning and proceeds - the machine writing it knows better than this
    function does. Over 65,536 px in either direction matplotlib fails
    outright, so that is capped rather than left to crash. Vector output has
    neither limit.
    """
    w, h = fig.get_size_inches()
    mpx = w * dpi * h * dpi / 1e6
    hard = 65000 / max(w, h)
    if dpi > hard:
        capped = max(72, int(hard))
        print(f"Note: {dpi} dpi would exceed matplotlib's 65,536 px limit on a "
              f"{w:.0f} x {h:.0f} in figure; using {capped} dpi. Write .svg "
              f"instead for unlimited resolution.")
        return capped
    if mpx > warn_mpx:
        print(f"Note: {dpi} dpi gives a {w*dpi:.0f} x {h*dpi:.0f} px image "
              f"({mpx:.0f} Mpx), which needs roughly {mpx*4:.0f} MB while it "
              f"is written. Use --dpi 150, fewer panels, or .svg if that is "
              f"too much.")
    return dpi


def compact_svg(path, precision=None):
    """
    Rewrite the rectangle subpaths matplotlib emitted into a shorter form.

    Matplotlib writes every rectangle as `M x0 y0 L x1 y0 L x1 y1 L x0 y1 z`
    with a newline after each command - eight numbers where four will do. The
    same corners as `M x0 y0 H x1 V y1 H x0 z` cost about 45% fewer characters,
    and no coordinate changes, so the geometry is untouched.

    `precision`, if given, also rounds coordinates to that many decimals. An
    axis holds a few thousand bases per point, so 4 decimals still resolves
    well under one base; it is off by default so the output is exactly what
    matplotlib computed.
    """
    def trim(s):
        if precision is None:
            return s
        return f"{round(float(s), precision):.{precision}f}".rstrip('0').rstrip('.') or '0'

    num = r'-?\d+(?:\.\d+)?'
    rect = re.compile(
        rf'M ({num}) ({num})\s*L ({num}) ({num})\s*L ({num}) ({num})\s*'
        rf'L ({num}) ({num})\s*z\s*')

    def repl(m):
        x0, y0, x1 = m.group(1), m.group(2), m.group(3)
        y1 = m.group(6)
        return f"M{trim(x0)} {trim(y0)}H{trim(x1)}V{trim(y1)}H{trim(x0)}z"

    with open(path) as f:
        svg = f.read()
    out, n = rect.subn(repl, svg)
    if n:
        with open(path, "w") as f:
            f.write(out)
    return len(svg), len(out), n


def save_figure(fig, out_path, args):
    """Write the figure, then shrink its path data when it is plain SVG."""
    fig.savefig(out_path,
                dpi=SAVE_DPI if VECTOR_OUTPUT else raster_dpi(fig, SAVE_DPI))
    if not args.no_compact and out_path.lower().endswith('.svg'):
        before, after, n = compact_svg(out_path, args.svg_precision)
        if n:
            print(f"Compacted {n:,} rectangle(s): {before/1e6:.2f} MB -> "
                  f"{after/1e6:.2f} MB ({100*(1-after/before):.0f}% smaller)"
                  + (f", coordinates rounded to {args.svg_precision} dp"
                     if args.svg_precision is not None else ", geometry unchanged"))
    print(f"Wrote {out_path}")
    plt.close(fig)


# ===========================================================================
# Drawing helpers
# ===========================================================================

def rects_path(rects):
    """
    One compound Path covering many rectangles, given (x0, y0, x1, y1) each.

    The SVG backend writes one <path> element per artist, so a track of a
    hundred thousand stretches drawn as a hundred thousand Rectangles becomes a
    hundred thousand elements, each carrying its own repeated style and
    clip-path attributes - tens of megabytes, and slow in any viewer because
    every element is a separate object to lay out. Collapsing rectangles that
    share a colour into a single Path keeps every coordinate exactly as it was
    while emitting one element for the lot.
    """
    r = np.asarray(rects, dtype=float)
    n = len(r)
    verts = np.empty((n * 5, 2))
    verts[0::5, 0], verts[0::5, 1] = r[:, 0], r[:, 1]
    verts[1::5, 0], verts[1::5, 1] = r[:, 2], r[:, 1]
    verts[2::5, 0], verts[2::5, 1] = r[:, 2], r[:, 3]
    verts[3::5, 0], verts[3::5, 1] = r[:, 0], r[:, 3]
    verts[4::5] = verts[0::5]
    codes = np.full(n * 5, Path.LINETO, dtype=Path.code_type)
    codes[0::5] = Path.MOVETO
    codes[4::5] = Path.CLOSEPOLY
    return Path(verts, codes)


def fit_yticks(ax, fontsize=7, two_sided=False):
    """
    Blank tick labels the axis is too short to show without them colliding.

    matplotlib puts ticks wherever it is asked and never checks whether the
    labels fit. With a hundred panels each axis is about 20 pt tall, and six
    ticks on it overlap by several points. Every tick mark is kept and as many
    labels as there is room for are shown.

    `two_sided` thins outward from zero rather than downward from the top,
    for the mirror axis, which runs both ways and would otherwise lose one end
    entirely. Call this after tight_layout, when the axis is its final height.
    """
    ticks = list(ax.get_yticks())
    labels = [t.get_text() for t in ax.get_yticklabels()]
    if len(ticks) < 2 or len(labels) != len(ticks):
        return
    lo, hi = ax.get_ylim()
    span = abs(hi - lo)
    if span <= 0:
        return
    h_pt = ax.get_position().height * ax.get_figure().get_size_inches()[1] * 72
    gap_pt = h_pt * abs(ticks[1] - ticks[0]) / span
    step = max(1, math.ceil((fontsize * 1.6) / max(gap_pt, 1e-6)))
    if two_sided:
        keep = {i for i, t in enumerate(ticks)
                if int(round(abs(t))) % step == 0}
        keep |= {0, len(ticks) - 1}
    else:
        keep = set(range(len(ticks) - 1, -1, -step))
    ax.set_yticklabels(
        [lab if i in keep else '' for i, lab in enumerate(labels)],
        fontsize=fontsize)


def draw_strips(ax, rows, x_max, centred=False):
    """
    One grey strip per region, behind everything.

    A region whose bar has no height - depth 1, or a fraction of 1 - would
    otherwise leave no mark at all, and a called region should always be
    locatable. `centred` straddles zero, for the mirror axis where the space
    below is in use.
    """
    min_w = min_draw_width(ax, x_max)
    extent = {}
    for row in rows:
        lo, hi = extent.get(row.region, (row.start, row.end))
        extent[row.region] = (min(lo, row.start), max(hi, row.end))
    if centred:
        strips = [(lo, -BASELINE_H / 2, lo + max(hi - lo, min_w), BASELINE_H / 2)
                  for lo, hi in extent.values()]
    else:
        strips = [(lo, -BASELINE_H, lo + max(hi - lo, min_w), 0.0)
                  for lo, hi in extent.values()]
    if strips:
        ax.add_patch(PathPatch(rects_path(strips), linewidth=0,
                               edgecolor='none',
                               facecolor='0.45' if centred else '0.55',
                               zorder=1))


def draw_bars(ax, bars, x_max, downward=False):
    """
    Bars given as (start, end, height, colour) with colour an "#rrggbb" string.

    Grouped by colour so the whole track is a few hundred SVG elements rather
    than one per stretch, however many distinct values it holds. Heights stay
    exact - a compound path carries a different height per rectangle - so only
    the grouping depends on colour, not the geometry.

    An outline only helps while a bar is wide enough to have an inside; below
    that it is all edge and the track turns into a picket fence, so narrow bars
    get none.
    """
    min_w = min_draw_width(ax, x_max)
    narrow = x_max / 250.0
    groups = defaultdict(list)
    for start, end, height, colour in bars:
        if height <= 0:
            continue
        wide = (end - start) > narrow
        y0, y1 = (-height, 0.0) if downward else (0.0, height)
        groups[(colour, wide)].append(
            (start, y0, start + max(end - start, min_w), y1))
    for (colour, wide), rects in sorted(groups.items()):
        ax.add_patch(PathPatch(
            rects_path(rects),
            linewidth=0.4 if wide else 0,
            edgecolor='white' if wide else 'none',
            facecolor=colour, zorder=2))


def draw_plasmid(ax, plasmid_region, y_low, y_high):
    """The known plasmid extent, as a pale band behind the track."""
    if plasmid_region is None:
        return
    p_start, p_end = plasmid_region
    ax.add_patch(patches.Rectangle(
        (p_start, y_low), p_end - p_start, y_high - y_low,
        linewidth=0, facecolor='tab:blue', alpha=0.13, zorder=0))


# ===========================================================================
# Labels
# ===========================================================================

def depth_y_high(max_depth, linear):
    """Top of a depth axis."""
    return (max_depth * 1.08 if linear
            else math.log2(max(2, max_depth)) + 0.35)


def fraction_y_high(heights, linear):
    """Top of a fraction axis, given the bar heights it must hold."""
    return 1.02 if linear else max(max(heights, default=0.0) + 0.35, 1.0)


def mirror_k(max_depth, drops, pinned=None):
    """
    (k, needed) for a mirror axis: k ticks each side, top 2**k, bottom 1/2**k.

    Taken as whichever half needs more, so the halves are the same size. `k`
    still moves with the data, so `pinned` is the way to get figures that are
    directly comparable across a set.
    """
    needed = max(1,
                 math.ceil(math.log2(max(2, max_depth))),
                 math.ceil(max(drops, default=0.0)))
    return (pinned if pinned else needed), needed


def clade_label(k):
    """Tick label for log2 clade count k: 1, 2, 4, 8..."""
    return str(2 ** k)


def fraction_label(k):
    """
    Tick label for -log2(fraction) k: 1, 1/2, 1/4, 1/8...

    Base two, matching the clade axis, so a unit is one doubling on either
    side of a mirror plot rather than a doubling above and a tenfold below.
    """
    return "1" if k <= 0 else f"1/{2 ** k}"


def pct(fraction):
    """A fraction as a short percentage string, for printed messages."""
    p = fraction * 100
    if p >= 1:
        return f"{p:.0f}%"
    if p >= 0.001:
        return f"{p:.2g}%"
    return f"{p:.0e}%".replace("e-0", "e-")


# ===========================================================================
# Input
# ===========================================================================

def is_known_name(q_name):
    """True when a Q name parses as plasmid_id,plasmid_start,plasmid_end,host_id."""
    parts = q_name.split(',')
    if len(parts) != 4:
        return False
    try:
        int(parts[1])
        int(parts[2])
    except ValueError:
        return False
    return True


def parse_fasta_lengths(fasta_path):
    """Return {seq_id: length} for every sequence in a FASTA file."""
    lengths = {}
    current_id, current_len = None, 0
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_id is not None:
                    lengths[current_id] = current_len
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id] = current_len
    return lengths


def detect_known(df, use_known):
    """
    Whether Q names follow the plasmid convention.

    Only used when every name follows it, so ordinary accessions fall back to
    plain grouping instead of silently scoring 0% against a 0-0 plasmid region.
    """
    if use_known is not None:
        return use_known
    names = df['Q name'].astype(str).unique()
    matching = sum(1 for n in names if is_known_name(n))
    use_known = len(names) > 0 and matching == len(names)
    style = 'plasmid_id,start,end,host_id' if use_known else 'plain'
    print(f"Q name format detected: {style} "
          f"({matching}/{len(names)} match the plasmid convention). "
          f"Override with --known or --no-known.")
    return use_known


def read_tsv(path, required):
    """Load a TSV and check the columns a mode needs are present."""
    if not os.path.exists(path):
        print(f"Error: no such file: {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path, sep='\t', comment='#')
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        print(f"Error: could not read {path} as a TSV: {exc}")
        sys.exit(1)
    df.columns = df.columns.str.strip()
    missing = set(required) - set(df.columns)
    if missing:
        print(f"Error: missing columns in {path}: {sorted(missing)}")
        print(f"Present: {sorted(df.columns)}")
        sys.exit(1)
    if df.empty:
        print(f"Error: {path} has a header but no rows.")
        sys.exit(1)
    return df


class Row:
    """One drawn stretch: an interval plus whatever quantities it carries."""
    __slots__ = ("start", "end", "depth", "fraction", "region", "hit", "avail")

    def __init__(self, start, end, depth, fraction, region, hit=0, avail=0):
        self.start, self.end = start, end
        self.depth, self.fraction, self.region = depth, fraction, region
        # Kept alongside the ratio for the interactive page, which reports the
        # counts and recomputes the fraction itself.
        self.hit, self.avail = hit, avail


def read_track(path, use_known, need_depth, need_fraction):
    """
    Group rows by query for the depth, fraction and mirror modes.

    Returns (tracks, plasmid_regions, q_sizes, use_known). A summary TSV has no
    Depth column, but Peak Clades is the same quantity per region, so either
    file works.
    """
    base = ['Q name', 'Q start', 'Q end']
    frac_cols = ['Tree Leaves Hit', 'Tree Leaves In LCA']
    df = read_tsv(path, base + (frac_cols if need_fraction else []))

    depth_col = None
    if need_depth:
        depth_col = next((c for c in ('Depth', 'Peak Clades', 'Num Clades')
                          if c in df.columns), None)
        if depth_col is None:
            print(f"Error: {path} has none of Depth, Peak Clades, Num Clades.")
            sys.exit(1)
        print(f"Reading clade counts from '{depth_col}'.")

    use_known = detect_known(df, use_known)
    has_qsize = 'Q size' in df.columns
    has_region = 'Region Index' in df.columns
    tracks, plasmid_regions, q_sizes = {}, {}, {}
    n_no_leaves = n_over = 0

    for _, row in df.iterrows():
        q_name = str(row['Q name'])
        if use_known and is_known_name(q_name):
            p_id, p_start, p_end, host_id = q_name.split(',')
            key = (host_id, p_id)
            plasmid_regions[key] = (int(p_start), int(p_end))
        else:
            key = (q_name, '')
        tracks.setdefault(key, [])
        if has_qsize and key not in q_sizes:
            q_sizes[key] = int(row['Q size'])

        fraction = None
        hit = avail = 0
        if need_fraction:
            hit = int(row['Tree Leaves Hit'])
            avail = int(row['Tree Leaves In LCA'])
            if hit <= 0 or avail <= 0:
                n_no_leaves += 1          # no tree-matched clades, no fraction
            else:
                fraction = hit / avail
                if hit > avail:
                    # Only reachable when a hit leaf is in the database index
                    # but absent from the tree the run used, so it counts in
                    # the numerator and not the denominator.
                    n_over += 1
                    fraction = 1.0
        tracks[key].append(Row(
            int(row['Q start']), int(row['Q end']),
            int(row[depth_col]) if depth_col else 0, fraction,
            int(row['Region Index']) if has_region else 0, hit, avail))

    if n_no_leaves:
        print(f"{n_no_leaves:,} row(s) have no tree-matched clades, so they "
              f"have no fraction and are drawn as a bare region strip.")
    if n_over:
        print(f"Warning: {n_over:,} row(s) hit more leaves than their LCA "
              f"contains, so a hit leaf is missing from the tree. Clamped to 1.")

    for key in tracks:
        tracks[key].sort(key=lambda r: (r.start, r.end))
    return tracks, plasmid_regions, q_sizes, use_known


# ===========================================================================
# Filtering and panel layout, shared by the track modes
# ===========================================================================

def apply_selection(tracks, args, has_depth, has_fraction):
    """
    Drop whole queries that no stretch of qualifies, then apply --only.

    A query is kept when ANY stretch of it passes, and its rows are left
    untouched, so the panels still tile their regions rather than showing
    holes where a filter bit.
    """
    all_keys = list(tracks.keys())
    if has_depth and args.min_depth:
        tracks = {k: v for k, v in tracks.items()
                  if any(r.depth >= args.min_depth for r in v)}
        print(f"{len(tracks)} of {len(all_keys)} query/queries reach "
              f"depth >= {args.min_depth}")
    if has_fraction and args.max_fraction is not None:
        before = len(tracks)
        tracks = {k: v for k, v in tracks.items()
                  if any(r.fraction is not None and r.fraction <= args.max_fraction
                         for r in v)}
        print(f"{len(tracks)} of {before} remaining query/queries reach a "
              f"fraction <= {args.max_fraction:g}")
    if args.only:
        invalid = args.only - set(range(1, len(all_keys) + 1))
        if invalid:
            print(f"Warning: out of range (max {len(all_keys)}): {sorted(invalid)}")
        keep = [all_keys[i - 1] for i in sorted(args.only)
                if 1 <= i <= len(all_keys)]
        tracks = {k: tracks[k] for k in keep if k in tracks}
    if not tracks:
        print("Nothing to plot.")
        sys.exit(1)
    return tracks, all_keys


def x_max_for(key, q_sizes, fasta_lengths):
    return q_sizes.get(key) or fasta_lengths.get(key[0], X_MAX_DEFAULT)


def new_figure(n, per_row_small, per_row_budget):
    """A figure sized for n stacked panels."""
    height_per_row = (per_row_small if n <= 8
                      else max(per_row_small * 0.4, per_row_budget / n))
    return plt.figure(figsize=(14, height_per_row * n + 1.0)), height_per_row


def style_axis(ax, label, x_max, y_low, y_high, ticks, tick_labels):
    x_step = max(1, x_max // 10)
    ax.set_ylabel(label, rotation=0, labelpad=26, va='center', fontsize=9)
    ax.set_xticks(range(0, x_max + x_step, x_step))
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    # Last, so neither the patches nor the ticks can widen the view.
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_low, y_high)


def add_colorbar(fig, rect, norm, cmap, ticks, labels, label):
    cax = fig.add_axes(rect)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       ticks=ticks)
    bar.ax.set_yticklabels(labels, fontsize=7)
    bar.ax.minorticks_off()
    bar.set_label(label, fontsize=8)
    bar.outline.set_linewidth(0.4)
    return bar


# ===========================================================================
# Mode: depth
# ===========================================================================

def depth_height(depth, linear):
    if depth < 1:
        return 0.0
    return float(depth) if linear else math.log2(depth)


def cmd_depth(args):
    tracks, plasmid_regions, q_sizes, known = read_track(
        args.input, args.known, need_depth=True, need_fraction=False)
    tracks = dict(sorted(tracks.items()))
    tracks, all_keys = apply_selection(tracks, args, True, False)

    max_depth = max((r.depth for rows in tracks.values() for r in rows),
                    default=2)
    y_high = depth_y_high(max_depth, args.linear)
    cmap = plt.get_cmap(DEPTH_CMAP)
    # Log, not linear. Depth doubles rather than climbs, so a linear ramp
    # crushes the interesting range: against a file whose deepest stretch is
    # 16, depths 2 and 4 both land in the palest fifth of the colours.
    norm = LogNorm(vmin=1, vmax=max(2, max_depth))

    n = len(tracks)
    print(f"Plotting {n} panel(s); deepest stretch has {max_depth} clade(s).")
    fig, _ = new_figure(n, 1.35, 55)
    axes = []
    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        ax = fig.add_subplot(n, 1, pos)
        axes.append(ax)
        x_max = x_max_for(key, q_sizes, args.fasta_lengths)
        draw_plasmid(ax, plasmid_regions.get(key) if known else None,
                     -BASELINE_H, y_high)
        draw_strips(ax, rows, x_max)
        draw_bars(ax, [(r.start, r.end, depth_height(r.depth, args.linear),
                        to_hex(cmap(norm(r.depth)))) for r in rows], x_max)
        if args.linear:
            step = max(1, int(y_high) // 4)
            ticks = [t for t in range(0, int(y_high) + 1) if t % step == 0]
            labels = [str(t) for t in ticks]
        else:
            ticks = list(range(0, int(math.floor(y_high)) + 1))
            labels = [clade_label(k) for k in ticks]
        style_axis(ax, str(all_keys.index(key) + 1), x_max,
                   -BASELINE_H, y_high, ticks, labels)
        ax.axhline(0, color='0.3', linewidth=0.6, zorder=3)
        ax.set_title(f'{key[0]} {key[1]}'.strip(), fontsize=9)

    fig.supylabel('Clades covering' + ('' if args.linear else ' (log$_2$)'),
                  fontsize=10)
    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Clade depth - {os.path.basename(args.input)}', fontsize=11)
    plt.tight_layout(rect=[0.02, 0.01, 0.925, 0.98])
    plt.subplots_adjust(hspace=0.85)

    bar_ticks = [2 ** k for k in range(0, int(math.floor(math.log2(
        max(2, max_depth)))) + 1)]
    if bar_ticks[-1] != max_depth:
        bar_ticks.append(max_depth)
    bar = add_colorbar(fig, [0.940, 0.12, 0.011, 0.74], norm, cmap,
                       bar_ticks, [str(t) for t in bar_ticks], 'Clades')
    for a in axes + [bar.ax]:
        fit_yticks(a)
    return fig


# ===========================================================================
# Mode: fraction
# ===========================================================================

def fraction_height(fraction, linear):
    """
    Bar height for a fraction, or 0 when there is nothing to draw.

    -log2(fraction) by default, so a fraction of 1 is flat and 1/16 stands four
    units tall. The interesting stretches are the small fractions, and this is
    what makes them the tall ones. --linear plots 1 - fraction instead, which
    keeps the same direction on a bounded axis.
    """
    if fraction is None or fraction <= 0:
        return 0.0
    fraction = min(1.0, fraction)
    return (1.0 - fraction) if linear else -math.log2(fraction)


def cmd_fraction(args):
    tracks, plasmid_regions, q_sizes, known = read_track(
        args.input, args.known, need_depth=False, need_fraction=True)
    tracks = dict(sorted(tracks.items()))
    tracks, all_keys = apply_selection(tracks, args, False, True)

    fracs = [r.fraction for rows in tracks.values() for r in rows
             if r.fraction is not None and 0 < r.fraction <= 1]
    if not fracs:
        print("No row has both a hit leaf and an LCA leaf count, so there is "
              "no fraction to draw.")
        sys.exit(1)
    heights = [fraction_height(f, args.linear) for f in fracs]
    y_high = fraction_y_high(heights, args.linear)

    cmap = plt.get_cmap(FRACTION_CMAP)
    norm = Normalize(vmin=0.0, vmax=y_high)

    n = len(tracks)
    print(f"Plotting {n} panel(s); smallest fraction {min(fracs):.3g} "
          f"({pct(min(fracs))} of its LCA subtree).")
    fig, _ = new_figure(n, 1.35, 55)
    axes = []
    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        ax = fig.add_subplot(n, 1, pos)
        axes.append(ax)
        x_max = x_max_for(key, q_sizes, args.fasta_lengths)
        draw_plasmid(ax, plasmid_regions.get(key) if known else None,
                     -BASELINE_H, y_high)
        draw_strips(ax, rows, x_max)
        bars = [(r.start, r.end, h, to_hex(cmap(norm(h))))
                for r in rows
                for h in (fraction_height(r.fraction, args.linear),)]
        draw_bars(ax, bars, x_max)
        if args.linear:
            ticks = [t / 4 for t in range(0, 5)]
            labels = [f"{(1 - t) * 100:.0f}%" for t in ticks]
        else:
            ticks = list(range(0, int(math.floor(y_high)) + 1))
            labels = [fraction_label(k) for k in ticks]
        style_axis(ax, str(all_keys.index(key) + 1), x_max,
                   -BASELINE_H, y_high, ticks, labels)
        ax.axhline(0, color='0.3', linewidth=0.6, zorder=3)
        ax.set_title(f'{key[0]} {key[1]}'.strip(), fontsize=9)

    fig.supylabel('Fraction of the LCA subtree hit'
                  + ('' if args.linear else ' (log$_2$)'), fontsize=10)
    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Clade breadth - {os.path.basename(args.input)}', fontsize=11)
    plt.tight_layout(rect=[0.02, 0.01, 0.925, 0.98])
    plt.subplots_adjust(hspace=0.85)

    if args.linear:
        bar_ticks = [t / 4 for t in range(0, 5)]
        bar_labels = [f"{(1 - t) * 100:.0f}%" for t in bar_ticks]
    else:
        bar_ticks = list(range(0, int(math.floor(y_high)) + 1))
        bar_labels = [fraction_label(k) for k in bar_ticks]
    bar = add_colorbar(fig, [0.940, 0.12, 0.011, 0.74], norm, cmap,
                       bar_ticks, bar_labels, 'Fraction hit')
    for a in axes + [bar.ax]:
        fit_yticks(a)
    return fig


# ===========================================================================
# Mode: mirror
# ===========================================================================

def cmd_mirror(args):
    tracks, plasmid_regions, q_sizes, known = read_track(
        args.input, args.known, need_depth=True, need_fraction=True)
    tracks = dict(sorted(tracks.items()))
    tracks, all_keys = apply_selection(tracks, args, True, True)

    depths = [r.depth for rows in tracks.values() for r in rows]
    fracs = [r.fraction for rows in tracks.values() for r in rows
             if r.fraction is not None and 0 < r.fraction < 1]
    max_depth = max(depths) if depths else 2
    drops = [-math.log2(f) for f in fracs] or [0.0]

    # One symmetric scale for both halves. The topmost clade tick is 2**k and
    # the lowest fraction tick is 1/2**k for the same k, taken as whichever
    # half needs more: k = max(ceil(log2(deepest)), ceil(max drop)). Both are
    # log2, so a unit is one doubling either way, and giving them the same
    # number of units makes the halves the same size.
    #
    # k still moves with the data, so two samples only match when they happen
    # to need the same k. --ticks pins it, which is the way to get figures that
    # are directly comparable across a whole set.
    k, needed = mirror_k(max_depth, drops, args.ticks)
    if args.ticks and needed > args.ticks:
        over_depth = sum(1 for d in depths if d > 2 ** args.ticks)
        over_frac = sum(1 for x in drops if x > args.ticks)
        print(f"Warning: --ticks {args.ticks} is below the {needed} this data "
              f"needs, so bars are clipped at the frame: {over_depth:,} "
              f"stretch(es) deeper than {2 ** args.ticks} clades and "
              f"{over_frac:,} with a fraction under 1/{2 ** args.ticks}.")
    y_top = y_bot = k + 0.2      # a little air so a full-height bar clears the
                                 # frame rather than merging into it

    depth_cmap = plt.get_cmap(DEPTH_CMAP)
    depth_norm = LogNorm(vmin=1, vmax=max(2, max_depth))
    frac_cmap = plt.get_cmap(FRACTION_CMAP)
    # Colour is normalised to the data, not to the axis: tying it to k would
    # wash out one half whenever the other half is what set k.
    frac_vmax = max(max(drops), 1.0)
    frac_norm = Normalize(vmin=0.0, vmax=frac_vmax)

    n = len(tracks)
    print(f"Plotting {n} panel(s); deepest stretch {max_depth} clade(s), "
          f"smallest fraction {pct(min(fracs)) if fracs else 'n/a'}; "
          f"axis 1..{2 ** k} up and 1..1/{2 ** k} down.")
    # Both halves are the same size now, so each gets half the panel and
    # neither needs the extra height a lopsided axis did.
    fig, _ = new_figure(n, 1.8, 75)
    axes = []
    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        ax = fig.add_subplot(n, 1, pos)
        axes.append(ax)
        x_max = x_max_for(key, q_sizes, args.fasta_lengths)
        draw_plasmid(ax, plasmid_regions.get(key) if known else None,
                     -y_bot, y_top)
        draw_strips(ax, rows, x_max, centred=True)
        draw_bars(ax, [(r.start, r.end, depth_height(r.depth, False),
                        to_hex(depth_cmap(depth_norm(r.depth))))
                       for r in rows if r.depth >= 2], x_max)
        down = [(r.start, r.end, h, to_hex(frac_cmap(frac_norm(h))))
                for r in rows
                for h in (fraction_height(r.fraction, False),) if h > 0]
        draw_bars(ax, down, x_max, downward=True)

        # Powers of two both ways, the same count each side. Zero is left
        # blank: it is depth 1 above the line and a fraction of 1 below it at
        # the same time.
        ticks = list(range(-k, k + 1))
        labels = ["" if k == 0 else
                  (clade_label(k) if k > 0 else fraction_label(-k))
                  for k in ticks]
        style_axis(ax, str(all_keys.index(key) + 1), x_max,
                   -y_bot, y_top, ticks, labels)
        ax.axhline(0, color='0.25', linewidth=0.7, zorder=3)
        ax.set_title(f'{key[0]} {key[1]}'.strip(), fontsize=9)

    # Two captions, not one: a rotated supylabel reads bottom-to-top, so a
    # single string would put the upward half's caption at the bottom of the
    # figure. Each gets about 40% of the figure height to run along, so on a
    # one-panel figure - under three inches tall - a 10 pt caption would be
    # longer than the space and the two would collide. Sized to fit.
    up_label = '\u2191 clades (log$_2$)'
    dn_label = '\u2193 fraction hit (log$_2$)'
    budget_pt = fig.get_size_inches()[1] * 72 * 0.42
    label_pt = max(5.5, min(10.0, budget_pt /
                            (max(len(up_label), len(dn_label)) * 0.58)))
    fig.text(0.012, 0.73, up_label, rotation=90, va='center', ha='center',
             fontsize=label_pt)
    fig.text(0.012, 0.27, dn_label, rotation=90, va='center', ha='center',
             fontsize=label_pt)
    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Clade depth and breadth - {os.path.basename(args.input)}',
                 fontsize=11)
    plt.tight_layout(rect=[0.035, 0.01, 0.915, 0.98])
    plt.subplots_adjust(hspace=0.9)

    up_ticks = [2 ** k for k in range(0, int(math.floor(math.log2(
        max(2, max_depth)))) + 1)]
    if up_ticks[-1] != max_depth:
        up_ticks.append(max_depth)
    bar_up = add_colorbar(fig, [0.935, 0.55, 0.011, 0.32], depth_norm,
                          depth_cmap, up_ticks,
                          [str(t) for t in up_ticks], 'Clades')
    dn_ticks = list(range(0, int(math.floor(frac_vmax)) + 1))
    bar_dn = add_colorbar(fig, [0.935, 0.13, 0.011, 0.32], frac_norm,
                          frac_cmap, dn_ticks,
                          [fraction_label(k) for k in dn_ticks], 'Fraction hit')
    for a in axes:
        fit_yticks(a, two_sided=True)
    for a in (bar_up.ax, bar_dn.ax):
        fit_yticks(a)
    return fig


# ===========================================================================
# Mode: area
# ===========================================================================

def compress(intervals):
    """
    Merge overlapping or adjacent intervals, carrying a 3rd element by max.

    Intervals are half-open [start, end), as alasight writes them: Q end is one
    past the last base, so contiguity is s == prev_end and NOT s == prev_end + 1
    - the closed-interval test, which fused regions separated by one uncovered
    base. That matters on real output: at the default --cluster-size 0 alasight
    leaves two regions exactly one base apart precisely because that base had
    fewer than two clades on it, and the merge carries the label by max, so the
    weaker region would inherit the stronger one's clade count.
    """
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for iv in sorted_ivs[1:]:
        s, e = iv[0], iv[1]
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            if len(iv) > 2 and len(merged[-1]) > 2:
                merged[-1][2] = max(merged[-1][2], iv[2])
        else:
            merged.append(list(iv))
    return [tuple(iv) for iv in merged]


def size_filter(intervals, min_size):
    """Remove intervals smaller than min_size bp."""
    return [iv for iv in intervals if (iv[1] - iv[0]) >= min_size]


def cluster_gap(intervals, gap):
    """Merge intervals within `gap` bp of each other, carrying 3rd by max."""
    if not intervals or gap <= 0:
        return intervals
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for iv in sorted_ivs[1:]:
        s, e = iv[0], iv[1]
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
            if len(iv) > 2 and len(merged[-1]) > 2:
                merged[-1][2] = max(merged[-1][2], iv[2])
        else:
            merged.append(list(iv))
    return [tuple(iv) for iv in merged]


def area_pipeline(intervals, min_size, gap):
    """compress, then size filter, then cluster - the order the plot uses."""
    return cluster_gap(size_filter(compress(intervals), min_size), gap)


def compute_score(raw_intervals, plasmid_regions, q_sizes, min_size, gap):
    """
    Detection metrics over the same intervals the figure draws.

    Compressing first is what keeps the numbers and the picture consistent:
    without it a pair of sub-threshold regions that compress fuses past
    --size-filter is drawn as a block while the score, filtering the unfused
    pair, reports 0% found. It also stops bases being counted twice when the
    input intervals overlap, as they do in a hits file rather than a summary.
    """
    metrics = []
    for key, (p_start, p_end) in plasmid_regions.items():
        # NOTE: p_start/p_end come from the Q name and are compared against
        # alasight coordinates, which are 0-based with an exclusive end. If
        # whatever generated those names used 1-based inclusive bounds then
        # mge_bp is short by one and every overlap is shifted by one.
        mge_bp = p_end - p_start
        host_bp = max(0, q_sizes.get(key, 0) - mge_bp)
        intervals = area_pipeline(raw_intervals.get(key, []), min_size, gap)
        bp_within = bp_outside = 0
        for iv in intervals:
            s, e = iv[0], iv[1]
            overlap = max(0, min(e, p_end) - max(s, p_start))
            bp_within += overlap
            bp_outside += (e - s) - overlap
        metrics.append({
            'coverage': min(1.0, bp_within / mge_bp) if mge_bp > 0 else 0,
            'bp_within': bp_within, 'bp_outside': bp_outside,
            'mge_bp': mge_bp, 'host_bp': host_bp,
        })
    if not metrics:
        return None
    found = [m for m in metrics if m['coverage'] > 0]
    total_host = sum(m['host_bp'] for m in metrics)
    total_mge = sum(m['mge_bp'] for m in found)
    return {
        'pct_found': len(found) / len(metrics),
        'pct_host': (sum(m['bp_outside'] for m in metrics) / total_host)
                    if total_host > 0 else 0,
        'pct_area': (sum(m['bp_within'] for m in found) / total_mge)
                    if total_mge > 0 else 0,
        'count_90': sum(1 for m in found if m['coverage'] > 0.9),
    }


def read_area(path, use_known):
    """
    Region intervals per query, plus the raw ones for scoring.

    Returns (regions, raw, plasmid_regions, q_sizes, label_col, use_known).
    """
    df = read_tsv(path, ['Q name', 'Q start', 'Q end'])
    use_known = detect_known(df, use_known)
    label_col = next((c for c in ('Peak Clades', 'Num Clades',
                                  'Num Unique Species') if c in df.columns), None)
    if label_col:
        print(f"Labelling regions with '{label_col}'.")

    has_qsize = 'Q size' in df.columns
    regions, raw, plasmid_regions, q_sizes = {}, {}, {}, {}
    bad_names = set()
    for _, row in df.iterrows():
        q_name = str(row['Q name'])
        q_start, q_end = int(row['Q start']), int(row['Q end'])
        if use_known:
            if not is_known_name(q_name):
                bad_names.add(q_name)
                p_id, p_start, p_end, host_id = "", 0, 0, q_name
            else:
                p_id, p_start, p_end, host_id = q_name.split(',')
                p_start, p_end = int(p_start), int(p_end)
            key = (host_id, p_id)
            plasmid_regions[key] = (p_start, p_end)
        else:
            key = (q_name, '')
        if key not in regions:
            regions[key], raw[key] = [], []
            if has_qsize:
                q_sizes[key] = int(row['Q size'])
        raw[key].append((q_start, q_end))
        regions[key].append((q_start, q_end, int(row[label_col]))
                            if label_col else (q_start, q_end))

    if bad_names:
        print(f"Warning: {len(bad_names)} Q name(s) do not match "
              f"plasmid_id,plasmid_start,plasmid_end,host_id, e.g. "
              f"{sorted(bad_names)[:3]}. Their plasmid region is recorded as "
              f"0-0, so they will score 0% found. Use --no-known if these are "
              f"ordinary names.")
    return regions, raw, plasmid_regions, q_sizes, label_col, use_known


def cmd_area(args):
    regions, raw, plasmid_regions, q_sizes, label_col, known = read_area(
        args.input, args.known)
    regions = dict(sorted(regions.items()))
    all_keys = list(regions.keys())

    regions = {k: area_pipeline(v, args.size_filter, args.cluster)
               for k, v in regions.items()}
    if args.size_filter or args.cluster:
        print(f"Applied: size filter >= {args.size_filter} bp, "
              f"cluster gap <= {args.cluster} bp")

    scores = compute_score(raw, plasmid_regions, q_sizes,
                           args.size_filter, args.cluster)
    if scores:
        print(f"\nScores ({os.path.basename(args.input)}):")
        print(f"  % MGEs found:          {scores['pct_found']*100:6.1f}%"
              f"   |  % host genome covered:  {scores['pct_host']*100:6.2f}%")
        print(f"  % area of found MGEs:  {scores['pct_area']*100:6.1f}%"
              f"   |  MGEs >90% found:          {scores['count_90']}")
    elif not q_sizes:
        print("\nNote: 'Q size' column not found - host coverage score "
              "unavailable.")

    if args.only:
        invalid = args.only - set(range(1, len(all_keys) + 1))
        if invalid:
            print(f"Warning: out of range (max {len(all_keys)}): {sorted(invalid)}")
        keep = [all_keys[i - 1] for i in sorted(args.only)
                if 1 <= i <= len(all_keys)]
        regions = {k: regions[k] for k in keep if k in regions}
    if not regions:
        print("No entries to plot.")
        sys.exit(1)

    n = len(regions)
    print(f"\nPlotting {n} panel(s).")
    fig, _ = new_figure(n, 1.5, 60)
    for pos, (key, ivs) in enumerate(regions.items(), start=1):
        ax = fig.add_subplot(n, 1, pos)
        x_max = x_max_for(key, q_sizes, args.fasta_lengths)
        draw_plasmid(ax, plasmid_regions.get(key) if known else None, 0, 1)
        # One path for the lot; they all share a colour. The regions reaching
        # here never overlap - compress guarantees it - so filling them in one
        # path paints the same pixels as filling them one by one. Where the
        # one-pixel floor widens two neighbours into contact, one path is in
        # fact the better rendering: a single alpha 0.5 fill rather than two
        # stacked into a darker spot that means nothing.
        min_w = min_draw_width(ax, x_max)
        rects, labels = [], []
        for iv in ivs:
            rects.append((iv[0], 0.0, iv[0] + max(iv[1] - iv[0], min_w), 1.0))
            if len(iv) > 2:
                labels.append(((iv[0] + iv[1]) / 2, iv[2]))
        if rects:
            ax.add_patch(PathPatch(rects_path(rects), linewidth=0,
                                   edgecolor='none', facecolor='red',
                                   alpha=0.5, zorder=2))
        for x_mid, ns in labels:
            ax.text(x_mid, 0.5, str(ns), ha='center', va='center',
                    fontsize=7, color='darkred', fontweight='bold', zorder=3)
        style_axis(ax, str(all_keys.index(key) + 1), x_max, 0, 1, [], [])
        ax.set_title(f'{key[0]} {key[1]}'.strip(), fontsize=9)

    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Detected regions - {os.path.basename(args.input)}',
                 fontsize=11)
    plt.tight_layout(rect=[0.02, 0.01, 1, 0.98])
    plt.subplots_adjust(hspace=0.8)
    return fig


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { --fg:#222; --muted:#666; --line:#bbb; --bg:#fff; }
  body { margin:0; padding:18px 20px 40px; background:var(--bg); color:var(--fg);
         font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  h1 { font-size:15px; font-weight:600; margin:0 0 2px; }
  .sub { color:var(--muted); margin:0 0 16px; font-size:12px; }
  .panel { margin:0 0 22px; }
  .head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin:0 0 4px; }
  .name { font-weight:600; }
  .range { color:var(--muted); font-variant-numeric:tabular-nums; font-size:12px; }
  canvas { display:block; width:100%; border:1px solid var(--line); cursor:crosshair;
           touch-action:none; background:var(--bg); }
  .ctl { display:flex; align-items:center; gap:16px; flex-wrap:wrap;
         margin:0 0 14px; color:var(--muted); font-size:12px; }
  button { font:inherit; padding:2px 9px; border:1px solid var(--line);
           background:#f6f6f6; border-radius:3px; cursor:pointer; color:var(--fg); }
  button:hover { background:#ececec; }
  label { display:flex; align-items:center; gap:5px; cursor:pointer; }
  .tip { position:fixed; pointer-events:none; background:rgba(20,20,20,.92);
         color:#fff; padding:6px 9px; border-radius:4px; font-size:12px;
         white-space:pre; display:none; z-index:10;
         font-variant-numeric:tabular-nums; }
  .legend { display:flex; gap:22px; flex-wrap:wrap; align-items:center;
            margin:0 0 16px; font-size:12px; color:var(--muted); }
  .ramp { display:inline-block; height:9px; width:120px; vertical-align:middle;
          border:1px solid var(--line); }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<div class="ctl">
  <span>Wheel to zoom x &middot; drag to pan &middot; double-click to reset</span>
  <label><input type="checkbox" id="boost"> Widen bars thinner than a pixel</label>
  <span id="boostnote"></span>
</div>
<div class="legend" id="legend"></div>
<div id="panels"></div>
<div class="tip" id="tip"></div>
<script>
const D = __PAYLOAD__;

/* Starts are delta-encoded, being sorted; undo that once. */
function undelta(a) {
  const out = new Float64Array(a.length);
  let acc = 0;
  for (let i = 0; i < a.length; i++) { acc += a[i]; out[i] = acc; }
  return out;
}
for (const p of D.panels) {
  p.starts = undelta(p.dstarts);
  delete p.dstarts;
  p.view = { x0: 0, x1: p.xMax };
}

const PAD = { l: 74, r: 14, t: 8, b: 26 };

/* Same mappings the static figures use, so a bar keeps its height and colour
   whatever the view: LogNorm(1, maxDepth) for depth, Normalize(0, fracVmax)
   for the fraction's drop. */
function depthHeight(depth) { return depth < 2 ? 0 : Math.log2(depth); }
function dropHeight(hit, avail) {
  if (!(hit > 0 && avail > 0)) return 0;
  const f = Math.min(1, hit / avail);
  return f >= 1 ? 0 : -Math.log2(f);
}
function pick(ramp, t) {
  return ramp[Math.max(0, Math.min(ramp.length - 1,
         Math.round(t * (ramp.length - 1))))];
}
function depthColor(depth) {
  if (!(D.maxDepth > 1)) return D.upRamp[0];
  return pick(D.upRamp, Math.log(Math.max(1, depth)) / Math.log(D.maxDepth));
}
function dropColor(drop) {
  return pick(D.downRamp, D.fracVmax > 0 ? drop / D.fracVmax : 0);
}
/* Which quantity a series carries decides its height AND its colour, so the
   fraction keeps its own ramp whether it is drawn above the axis (fraction
   mode) or below it (mirror). Inferring direction from the mode is what put
   the fraction below the axis in a mode whose axis has no room there. */
function seriesHeight(kind, p, i) {
  return kind === "depth" ? depthHeight(p.depths[i])
                          : dropHeight(p.hits[i], p.avails[i]);
}
function seriesColor(kind, p, i, h) {
  return kind === "depth" ? depthColor(p.depths[i]) : dropColor(h);
}

function niceTicks(x0, x1, want) {
  const span = x1 - x0;
  if (!(span > 0)) return [];
  const raw = span / want;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = mag;
  for (const m of [1, 2, 2.5, 5, 10]) if (mag * m >= raw) { step = mag * m; break; }
  const out = [];
  for (let v = Math.ceil(x0 / step) * step; v <= x1; v += step) out.push(v);
  return out;
}
function bp(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 1 : 2) + " Mb";
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + " kb";
  return Math.round(v) + " bp";
}

/* First index whose stretch can still reach x: starts are sorted, so binary
   search the left edge and walk forward. Only the visible slice is drawn, which
   is what keeps a zoomed view fast on a track of a hundred thousand stretches. */
function firstFrom(starts, x) {
  let lo = 0, hi = starts.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (starts[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

function draw(p) {
  const cv = p.canvas, ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = p.cssH;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const plotW = cssW - PAD.l - PAD.r, plotH = cssH - PAD.t - PAD.b;
  const span = D.yTop + D.yBot;
  const X = v => PAD.l + (v - p.view.x0) / (p.view.x1 - p.view.x0) * plotW;
  const Y = v => PAD.t + (D.yTop - v) / span * plotH;
  const minW = document.getElementById("boost").checked ? 1 / dpr : 0;

  /* Known plasmid extent, behind everything. */
  if (p.plasmid) {
    ctx.fillStyle = "rgba(31,119,180,0.13)";
    const a = X(p.plasmid[0]), b = X(p.plasmid[1]);
    ctx.fillRect(a, Y(D.yTop), b - a, Y(-D.yBot) - Y(D.yTop));
  }

  /* Called-region strips, so a region with no bar either way still shows. */
  ctx.fillStyle = "#737373";
  const h0 = Y(0), sh = Math.max(1, (D.baselineH / span) * plotH);
  for (const [rs, re] of p.regions) {
    if (re < p.view.x0 || rs > p.view.x1) continue;
    const a = X(rs), b = X(re);
    ctx.fillRect(a, h0 - sh / 2, Math.max(b - a, minW), sh);
  }

  /* Bars at their true width. Canvas antialiases a sub-pixel rectangle to
     partial coverage and accumulates it across overlapping draws, so a dense
     view reads as density rather than vanishing - and it does so identically
     in every browser, unlike a thin SVG path. */
  let drawn = 0;
  const i0 = firstFrom(p.starts, p.view.x0 - p.maxLen);
  for (let i = i0; i < p.starts.length; i++) {
    const s = p.starts[i];
    if (s > p.view.x1) break;
    const e = s + p.lens[i];
    if (e < p.view.x0) continue;
    const x = X(s), w = Math.max(X(e) - x, minW);
    if (D.upSeries) {
      const hh = seriesHeight(D.upSeries, p, i);
      if (hh > 0) {
        ctx.fillStyle = seriesColor(D.upSeries, p, i, hh);
        ctx.fillRect(x, Y(hh), w, h0 - Y(hh));
      }
    }
    if (D.downSeries) {
      const dd = seriesHeight(D.downSeries, p, i);
      if (dd > 0) {
        ctx.fillStyle = seriesColor(D.downSeries, p, i, dd);
        ctx.fillRect(x, h0, w, Y(-dd) - h0);
      }
    }
    drawn++;
  }

  /* Axes last, over the data. */
  ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.l, h0); ctx.lineTo(cssW - PAD.r, h0); ctx.stroke();
  ctx.strokeStyle = "#bbb";
  ctx.strokeRect(PAD.l, PAD.t, plotW, plotH);

  ctx.fillStyle = "#333"; ctx.font = "10px sans-serif";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of D.yTicks) {
    const y = Y(t.v);
    if (y < PAD.t - 1 || y > PAD.t + plotH + 1) continue;
    ctx.strokeStyle = "#ccc";
    ctx.beginPath(); ctx.moveTo(PAD.l - 3, y); ctx.lineTo(PAD.l, y); ctx.stroke();
    if (t.label) ctx.fillText(t.label, PAD.l - 6, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of niceTicks(p.view.x0, p.view.x1, 8)) {
    const x = X(v);
    ctx.strokeStyle = "#ccc";
    ctx.beginPath(); ctx.moveTo(x, PAD.t + plotH); ctx.lineTo(x, PAD.t + plotH + 3); ctx.stroke();
    ctx.fillStyle = "#333";
    ctx.fillText(bp(v), x, PAD.t + plotH + 5);
  }

  const perPx = (p.view.x1 - p.view.x0) / plotW;
  p.readout.textContent =
    bp(p.view.x0) + " \u2013 " + bp(p.view.x1) +
    "  (" + bp(p.view.x1 - p.view.x0) + " wide, " +
    (perPx < 1 ? (1 / perPx).toFixed(0) + " px/bp" : perPx.toFixed(0) + " bp/px") +
    ", " + drawn.toLocaleString() + " stretch" + (drawn === 1 ? "" : "es") + " in view)";
}

function clampView(p) {
  const minSpan = 50;
  if (p.view.x1 - p.view.x0 < minSpan) {
    const c = (p.view.x0 + p.view.x1) / 2;
    p.view.x0 = c - minSpan / 2; p.view.x1 = c + minSpan / 2;
  }
  if (p.view.x1 - p.view.x0 > p.xMax) { p.view.x0 = 0; p.view.x1 = p.xMax; return; }
  if (p.view.x0 < 0) { p.view.x1 -= p.view.x0; p.view.x0 = 0; }
  if (p.view.x1 > p.xMax) { p.view.x0 -= p.view.x1 - p.xMax; p.view.x1 = p.xMax; }
  if (p.view.x0 < 0) p.view.x0 = 0;
}

const tip = document.getElementById("tip");

function attach(p) {
  const cv = p.canvas;
  const toBp = ev => {
    const r = cv.getBoundingClientRect();
    const plotW = r.width - PAD.l - PAD.r;
    const frac = (ev.clientX - r.left - PAD.l) / plotW;
    return p.view.x0 + frac * (p.view.x1 - p.view.x0);
  };
  cv.addEventListener("wheel", ev => {
    ev.preventDefault();
    const at = toBp(ev);
    const f = Math.pow(1.0018, ev.deltaY);
    p.view.x0 = at - (at - p.view.x0) * f;
    p.view.x1 = at + (p.view.x1 - at) * f;
    clampView(p); draw(p);
  }, { passive: false });

  let dragging = null;
  cv.addEventListener("pointerdown", ev => {
    dragging = { bp: toBp(ev), x0: p.view.x0, x1: p.view.x1 };
    cv.setPointerCapture(ev.pointerId);
  });
  cv.addEventListener("pointerup", () => { dragging = null; });
  cv.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  cv.addEventListener("pointermove", ev => {
    if (dragging) {
      const r = cv.getBoundingClientRect();
      const plotW = r.width - PAD.l - PAD.r;
      const perPx = (dragging.x1 - dragging.x0) / plotW;
      const at = dragging.x0 + (ev.clientX - r.left - PAD.l) * perPx;
      const shift = dragging.bp - at;
      p.view.x0 = dragging.x0 + shift; p.view.x1 = dragging.x1 + shift;
      clampView(p); draw(p);
      return;
    }
    /* Nearest stretch under the cursor, within a pixel. */
    const at = toBp(ev);
    const r = cv.getBoundingClientRect();
    const perPx = (p.view.x1 - p.view.x0) / (r.width - PAD.l - PAD.r);
    let best = -1, bestD = perPx * 2;
    for (let i = firstFrom(p.starts, at - p.maxLen); i < p.starts.length; i++) {
      const s = p.starts[i];
      if (s > at + bestD) break;
      const e = s + p.lens[i];
      const d = at < s ? s - at : (at > e ? at - e : 0);
      if (d < bestD) { bestD = d; best = i; if (d === 0) break; }
    }
    if (best < 0) { tip.style.display = "none"; return; }
    const parts = [bp(p.starts[best]) + " \u2013 " + bp(p.starts[best] + p.lens[best]) +
                   "  (" + p.lens[best].toLocaleString() + " bp)"];
    if (D.upSeries === "depth" || D.downSeries === "depth")
      parts.push("clades: " + p.depths[best].toLocaleString());
    if (D.upSeries === "drop" || D.downSeries === "drop") {
      const h = p.hits[best], a = p.avails[best];
      parts.push(h > 0 && a > 0
        ? "leaves hit: " + h.toLocaleString() + " of " + a.toLocaleString() +
          "  (1/" + (a / h).toFixed(a / h >= 10 ? 0 : 1) + ")"
        : "leaves hit: none in the tree");
    }
    tip.textContent = parts.join("\n");
    tip.style.display = "block";
    tip.style.left = Math.min(window.innerWidth - 220, ev.clientX + 12) + "px";
    tip.style.top = (ev.clientY + 14) + "px";
  });
  cv.addEventListener("dblclick", () => {
    p.view.x0 = 0; p.view.x1 = p.xMax; draw(p);
  });
}

/* Build the page. */
const host = document.getElementById("panels");
D.panels.forEach((p, i) => {
  p.cssH = D.panelHeight;
  /* A loop, not Math.max(...arr) or .apply: those pass every element as an
     argument, and a track of a hundred thousand stretches overflows the
     call stack outright. */
  p.maxLen = 0;
  for (let i = 0; i < p.lens.length; i++) if (p.lens[i] > p.maxLen) p.maxLen = p.lens[i];
  const wrap = document.createElement("div");
  wrap.className = "panel";
  const head = document.createElement("div");
  head.className = "head";
  const nm = document.createElement("span");
  nm.className = "name"; nm.textContent = (i + 1) + ". " + p.name;
  const ro = document.createElement("span");
  ro.className = "range";
  const rst = document.createElement("button");
  rst.textContent = "Reset";
  rst.onclick = () => { p.view.x0 = 0; p.view.x1 = p.xMax; draw(p); };
  head.append(nm, ro, rst);
  const cv = document.createElement("canvas");
  wrap.append(head, cv);
  host.append(wrap);
  p.canvas = cv; p.readout = ro;
  attach(p);
});

/* Colour keys, matching the static figures' colourbars. */
function rampSwatch(ramp, lo, hi, label) {
  const c = document.createElement("canvas");
  c.width = 120; c.height = 9; c.className = "ramp";
  const g = c.getContext("2d");
  for (let x = 0; x < 120; x++) {
    g.fillStyle = ramp[Math.round(x / 119 * (ramp.length - 1))];
    g.fillRect(x, 0, 1, 9);
  }
  const box = document.createElement("span");
  box.append(document.createTextNode(label + " " + lo + " "), c,
             document.createTextNode(" " + hi));
  return box;
}
const legend = document.getElementById("legend");
if (D.upSeries === "depth" || D.downSeries === "depth")
  legend.append(rampSwatch(D.upRamp, "1", String(D.maxDepth), "clades"));
if (D.upSeries === "drop" || D.downSeries === "drop")
  legend.append(rampSwatch(D.downRamp, "1", D.fracLabel, "fraction hit"));

document.getElementById("boost").onchange = ev => {
  document.getElementById("boostnote").textContent = ev.target.checked
    ? "On: a stretch narrower than a pixel is drawn one pixel wide, so it is visible but wider than it is."
    : "";
  D.panels.forEach(draw);
};
window.addEventListener("resize", () => D.panels.forEach(draw));
D.panels.forEach(draw);
</script>
</body>
</html>
"""


def ramp_hex(cmap, n=64):
    """A colour ramp as hex, sampled from the same colormap the figures use."""
    from matplotlib.colors import to_hex
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def delta(values):
    """Sorted ints as first-order differences, which JSON stores far shorter."""
    out, prev = [], 0
    for v in values:
        out.append(int(v) - prev)
        prev = int(v)
    return out


def build_payload(tracks, all_keys, plasmid_regions, q_sizes, fasta_lengths,
                  known, mode, y_top, y_bot, y_ticks, base_h,
                  max_depth, frac_vmax, frac_label, up_cmap, down_cmap,
                  panel_height):
    """Everything the page needs, with the y axis already decided here."""
    panels = []
    for key, rows in tracks.items():
        x_max = q_sizes.get(key) or fasta_lengths.get(key[0], 100000)
        extent = {}
        for r in rows:
            lo, hi = extent.get(r.region, (r.start, r.end))
            extent[r.region] = (min(lo, r.start), max(hi, r.end))
        panels.append({
            "name": f"{key[0]} {key[1]}".strip(),
            "xMax": int(x_max),
            "plasmid": list(plasmid_regions[key]) if (known and key in plasmid_regions) else None,
            "dstarts": delta(r.start for r in rows),
            "lens": [int(r.end - r.start) for r in rows],
            "depths": [int(r.depth) for r in rows],
            "hits": [int(r.hit) for r in rows],
            "avails": [int(r.avail) for r in rows],
            "regions": [[int(a), int(b)] for a, b in sorted(extent.values())],
        })
    return {
        "mode": mode,
        # Which quantity goes above the axis and which below. The fraction is
        # drawn upward in its own mode and downward only in the mirror.
        "upSeries": {"depth": "depth", "fraction": "drop",
                     "mirror": "depth"}[mode],
        "downSeries": "drop" if mode == "mirror" else None,
        "yTop": y_top, "yBot": y_bot, "baselineH": base_h,
        "yTicks": y_ticks,
        "maxDepth": int(max_depth), "fracVmax": frac_vmax,
        "fracLabel": frac_label,
        # In fraction mode the upward series IS the fraction, so both ramps
        # are the fraction's and the legend shows one key.
        "upRamp": ramp_hex(down_cmap if mode == "fraction" else up_cmap),
        "downRamp": ramp_hex(down_cmap),
        "panelHeight": panel_height,
        "panels": panels,
    }


def cmd_html(args):
    """
    The interactive page for a track mode.

    Deliberately shares depth_y_high, fraction_y_high, mirror_k and the tick
    label helpers with the static figures: the y axis is fixed here, once, so
    a stretch is the same height and colour in both views.
    """
    need_depth = args.mode in ("depth", "mirror")
    need_frac = args.mode in ("fraction", "mirror")
    tracks, plasmid_regions, q_sizes, known = read_track(
        args.input, args.known, need_depth, need_frac)
    tracks = dict(sorted(tracks.items()))
    tracks, all_keys = apply_selection(tracks, args, need_depth, need_frac)

    depths = [r.depth for rows in tracks.values() for r in rows]
    fracs = [r.fraction for rows in tracks.values() for r in rows
             if r.fraction is not None and 0 < r.fraction < 1]
    max_depth = max(depths) if depths else 2
    drops = [-math.log2(f) for f in fracs]

    up_cmap = plt.get_cmap(DEPTH_CMAP)
    down_cmap = plt.get_cmap(FRACTION_CMAP)

    if args.mode == "mirror":
        k, needed = mirror_k(max_depth, drops, args.ticks)
        if args.ticks and needed > args.ticks:
            print(f"Warning: --ticks {args.ticks} is below the {needed} this "
                  f"data needs, so bars are clipped at the frame.")
        y_top = y_bot = k + 0.2
        frac_vmax = max(max(drops, default=0.0), 1.0)
        ticks = [{"v": t,
                  "label": "" if t == 0 else (clade_label(t) if t > 0
                                              else fraction_label(-t))}
                 for t in range(-k, k + 1)]
        frac_label = fraction_label(int(math.floor(frac_vmax)))
        panel_h = 240
        print(f"Axis 1..{2 ** k} up and 1..1/{2 ** k} down.")
    elif args.mode == "depth":
        y_top = depth_y_high(max_depth, False)
        y_bot = BASELINE_H
        frac_vmax, frac_label = 1.0, "1"
        ticks = [{"v": t, "label": clade_label(t)}
                 for t in range(0, int(math.floor(y_top)) + 1)]
        panel_h = 170
    else:
        heights = [fraction_height(f, False) for f in fracs]
        y_top = fraction_y_high(heights, False)
        y_bot = BASELINE_H
        frac_vmax = y_top
        ticks = [{"v": t, "label": fraction_label(t)}
                 for t in range(0, int(math.floor(y_top)) + 1)]
        frac_label = fraction_label(int(math.floor(y_top)))
        panel_h = 170

    payload = build_payload(
        tracks, all_keys, plasmid_regions, q_sizes, args.fasta_lengths, known,
        args.mode, y_top, y_bot, ticks, BASELINE_H, max_depth, frac_vmax,
        frac_label, up_cmap, down_cmap, panel_h)

    rows_total = sum(len(v) for v in tracks.values())
    title = f"alasight {args.mode} - {os.path.basename(args.input)}"
    sub = (f"{len(tracks)} panel(s), {rows_total:,} stretch(es). The y axis is "
           f"fixed; only x zooms, so a bar keeps its height and colour at every "
           f"scale. Bars are drawn at their true width, which is far under a "
           f"pixel until you zoom in.")
    size = write_html(payload, args.output, title, sub)
    print(f"Wrote {args.output} ({size/1e6:.2f} MB, {rows_total:,} stretches, "
          f"{len(tracks)} panel(s))")


def write_html(payload, path, title, subtitle):
    body = (HTML_TEMPLATE
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle))
    with open(path, "w") as f:
        f.write(body)
    return len(body)


# ===========================================================================
# CLI
# ===========================================================================

def comma_ints(text):
    try:
        return set(int(x) for x in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers, e.g. 3,7,12")


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", help="a TSV written by alasight")
    common.add_argument("output",
                        help="output file; SVG unless the extension says "
                             "otherwise, and .svg is added when absent")
    common.add_argument("--fasta",
                        help="FASTA to take sequence lengths from, when the "
                             "TSV has no 'Q size' column")
    common.add_argument("--only", type=comma_ints, default=set(),
                        help="plot only these 1-based panel numbers, e.g. 3,7,12")
    common.add_argument("--known", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="treat Q names as "
                             "plasmid_id,start,end,host_id (default: detect)")
    common.add_argument("--dpi", type=int, default=SAVE_DPI,
                        help=f"raster dpi (default: {SAVE_DPI}); ignored for "
                             f"vector output")
    common.add_argument("--no-compact", action="store_true",
                        help="skip the SVG path-data rewrite")
    common.add_argument("--svg-precision", type=int, default=None,
                        help="also round SVG coordinates to this many decimals")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    area = sub.add_parser("area", parents=[common],
                          help="shade every region, with its clade count on it")
    area.add_argument("--size-filter", type=int, default=0,
                      help="drop regions shorter than this many bp")
    area.add_argument("--cluster", type=int, default=0,
                      help="merge regions within this many bp of each other")
    area.add_argument("--all", action="store_true",
                      help="accepted for compatibility; all panels are always "
                           "drawn into one figure")
    area.set_defaults(func=cmd_area)

    depth = sub.add_parser("depth", parents=[common],
                           help="clade depth as a bar track")
    depth.add_argument("--min-depth", type=int, default=0,
                       help="only plot queries reaching this depth somewhere")
    depth.add_argument("--linear", action="store_true",
                       help="plot raw clade counts instead of log2")
    depth.set_defaults(func=cmd_depth)

    frac = sub.add_parser("fraction", parents=[common],
                          help="fraction of the LCA subtree hit")
    frac.add_argument("--max-fraction", type=float, default=None,
                      help="only plot queries reaching a fraction at or below "
                           "this somewhere")
    frac.add_argument("--linear", action="store_true",
                      help="plot 1 - fraction instead of -log2(fraction)")
    frac.set_defaults(func=cmd_fraction)

    mirror = sub.add_parser("mirror", parents=[common],
                            help="depth above the axis, breadth below it")
    mirror.add_argument("--min-depth", type=int, default=0,
                        help="only plot queries reaching this depth somewhere")
    mirror.add_argument("--max-fraction", type=float, default=None,
                        help="only plot queries reaching a fraction at or "
                             "below this somewhere")
    mirror.add_argument("--ticks", type=int, default=None,
                        help="pin the axis to this many ticks each side, so "
                             "the top tick is 2**N and the bottom 1/2**N. "
                             "Default: whichever half needs more. Pin it to "
                             "compare figures across samples; bars beyond it "
                             "are clipped and counted in a warning")
    mirror.set_defaults(func=cmd_mirror)
    return parser


def main(argv=None):
    global SAVE_DPI, VECTOR_OUTPUT
    args = build_parser().parse_args(argv)
    SAVE_DPI = args.dpi
    args.output, VECTOR_OUTPUT = resolve_output(args.output)

    if args.output.lower().endswith((".html", ".htm")):
        if args.mode == "area":
            print("Error: the interactive page covers the depth, fraction and "
                  "mirror modes; area writes an image.")
            sys.exit(1)
        args.fasta_lengths = {}
        if args.fasta:
            args.fasta_lengths = parse_fasta_lengths(args.fasta)
            print(f"Loaded lengths for {len(args.fasta_lengths)} sequences "
                  f"from FASTA.")
        cmd_html(args)
        return

    if VECTOR_OUTPUT:
        print(f"Writing {args.output}: vector, so regions are drawn at true "
              f"width with no one-pixel floor. Small ones are faint until you "
              f"zoom in.")
    else:
        print(f"Writing {args.output}: raster, so a region narrower than a "
              f"pixel is widened to one pixel to stay visible. Use an .svg "
              f"name for exact widths.")

    args.fasta_lengths = {}
    if args.fasta:
        args.fasta_lengths = parse_fasta_lengths(args.fasta)
        print(f"Loaded lengths for {len(args.fasta_lengths)} sequences from "
              f"FASTA.")
    if getattr(args, "all", False):
        print("Note: --all is a no-op; every panel goes into one figure.")

    fig = args.func(args)
    save_figure(fig, args.output, args)


if __name__ == "__main__":
    main()
