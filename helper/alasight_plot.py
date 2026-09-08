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
    from matplotlib.colors import LogNorm, Normalize
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

# Depth fixes both a bar's height and its colour, so bars can be grouped by it
# and drawn as one path each. A fraction is continuous, so its COLOUR is
# quantised into this many steps to restore the grouping. Heights stay exact;
# only the shade is rounded, and a sequential ramp does not resolve more steps
# than this by eye.
COLOR_BINS = 32

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


def draw_bars(ax, bars, x_max, cmap, downward=False):
    """
    Bars given as (start, end, height, colour_key, colour_position).

    Grouped by colour so the whole track is a handful of SVG elements rather
    than one per stretch. An outline only helps while a bar is wide enough to
    have an inside; below that it is all edge and the track turns into a picket
    fence, so narrow bars get none.
    """
    min_w = min_draw_width(ax, x_max)
    narrow = x_max / 250.0
    groups = defaultdict(list)
    for start, end, height, key, pos in bars:
        if height <= 0:
            continue
        wide = (end - start) > narrow
        y0, y1 = (-height, 0.0) if downward else (0.0, height)
        groups[(key, wide, pos)].append(
            (start, y0, start + max(end - start, min_w), y1))
    for (_key, wide, pos), rects in sorted(groups.items()):
        ax.add_patch(PathPatch(
            rects_path(rects),
            linewidth=0.4 if wide else 0,
            edgecolor='white' if wide else 'none',
            facecolor=cmap(pos), zorder=2))


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
    __slots__ = ("start", "end", "depth", "fraction", "region")

    def __init__(self, start, end, depth, fraction, region):
        self.start, self.end = start, end
        self.depth, self.fraction, self.region = depth, fraction, region


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
            int(row['Region Index']) if has_region else 0))

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
    y_high = (max_depth * 1.08 if args.linear
              else math.log2(max(2, max_depth)) + 0.35)
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
                        r.depth, norm(r.depth)) for r in rows],
                  x_max, cmap)
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
    y_high = 1.02 if args.linear else max(max(heights) + 0.35, 1.0)

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
        bars = []
        for r in rows:
            h = fraction_height(r.fraction, args.linear)
            shade = int(round(norm(h) * (COLOR_BINS - 1))) if h > 0 else 0
            bars.append((r.start, r.end, h, shade, shade / (COLOR_BINS - 1)))
        draw_bars(ax, bars, x_max, cmap)
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

    # Neither half is rescaled: both are log2, so one unit is one doubling
    # above the line and one doubling below it, and the halves are directly
    # comparable. The lower one usually runs deeper - a fraction of 1/500 is
    # nine doublings while a depth of 8 is three - and that is the honest
    # shape of the data rather than something to normalise away.
    y_top = math.log2(max(2, max_depth)) + 0.35
    y_bot = max(drops) + 0.35

    depth_cmap = plt.get_cmap(DEPTH_CMAP)
    depth_norm = LogNorm(vmin=1, vmax=max(2, max_depth))
    frac_cmap = plt.get_cmap(FRACTION_CMAP)
    frac_norm = Normalize(vmin=0.0, vmax=y_bot)

    n = len(tracks)
    print(f"Plotting {n} panel(s); deepest stretch {max_depth} clade(s), "
          f"smallest fraction {pct(min(fracs)) if fracs else 'n/a'}.")
    # Taller than the other modes: with both halves on one log2 scale the
    # lower one runs several times deeper, so the upward half needs the extra
    # absolute height to stay readable.
    fig, _ = new_figure(n, 2.2, 90)
    axes = []
    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        ax = fig.add_subplot(n, 1, pos)
        axes.append(ax)
        x_max = x_max_for(key, q_sizes, args.fasta_lengths)
        draw_plasmid(ax, plasmid_regions.get(key) if known else None,
                     -y_bot, y_top)
        draw_strips(ax, rows, x_max, centred=True)
        draw_bars(ax, [(r.start, r.end, depth_height(r.depth, False),
                        r.depth, depth_norm(r.depth))
                       for r in rows if r.depth >= 2],
                  x_max, depth_cmap)
        down = []
        for r in rows:
            h = fraction_height(r.fraction, False)
            if h <= 0:
                continue
            shade = int(round(frac_norm(h) * (COLOR_BINS - 1)))
            down.append((r.start, r.end, h, shade, shade / (COLOR_BINS - 1)))
        draw_bars(ax, down, x_max, frac_cmap, downward=True)

        # Powers of two both ways. Zero is left blank: it is depth 1 above the
        # line and a fraction of 1 below it at the same time.
        ticks = list(range(-int(math.floor(y_bot)), int(math.floor(y_top)) + 1))
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
    dn_ticks = list(range(0, int(math.floor(y_bot)) + 1))
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
    mirror.set_defaults(func=cmd_mirror)
    return parser


def main(argv=None):
    global SAVE_DPI, VECTOR_OUTPUT
    args = build_parser().parse_args(argv)
    SAVE_DPI = args.dpi
    args.output, VECTOR_OUTPUT = resolve_output(args.output)
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
