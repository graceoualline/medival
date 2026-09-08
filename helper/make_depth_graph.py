#!/usr/bin/env python3
# Draws the *_depth.tsv written by alasight as a clade-depth track: one panel
# per query, bar height = log2(clades covering that stretch).
#
# make_area_graph.py shades a region wherever two or more clades were found.
# This plots the number itself, so a locus with eight clades stacked on it
# stands a step above one where two brushed past each other.
#
#   python3 make_depth_graph.py <path_to_depth.tsv> <output.png>
#   python3 make_depth_graph.py depth.tsv out.png --only 3,7,12
#   python3 make_depth_graph.py depth.tsv out.png --linear
#   python3 make_depth_graph.py depth.tsv out.png --min-depth 3
#
# Flags mirror make_area_graph.py where they overlap.

import re
import sys
import os
from collections import defaultdict

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
except ModuleNotFoundError:
    print("Error: Matplotlib is not installed.")
    print("Please install Matplotlib by running: pip install matplotlib")
    sys.exit(1)
import math


def compact_svg(path, precision=None):
    """
    Rewrite the rectangle subpaths matplotlib emitted into a shorter form.

    Matplotlib writes every rectangle as `M x0 y0 L x1 y0 L x1 y1 L x0 y1 z`
    with a newline after each command - eight numbers where four will do. The
    same corners as `M x0 y0 H x1 V y1 H x0 z` cost about 45% fewer characters,
    and no coordinate changes, so the geometry is untouched.

    `precision`, if given, also rounds coordinates to that many decimals. The
    axis holds a few thousand bases per point, so 4 decimals still resolves
    well under one base; it is off by default so the default output is exactly
    what matplotlib computed.
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


def raster_dpi(fig, dpi, warn_mpx=60):
    """
    The dpi to actually use, given how large the raster would be.

    Panels stack, so a hundred-query run is a 14 x 56 in figure: 4,200 x 16,800
    px at 300 dpi, around 280 MB held in memory while it is written. That is
    merely large, so it gets a warning and proceeds - the machine writing it
    knows better than this function does. Over 65,536 px in either direction
    matplotlib fails outright, so that is capped rather than left to crash.
    Vector output has neither limit.
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


def fit_yticks(ax, fontsize=7):
    """
    Blank tick labels the axis is too short to show without them colliding.

    matplotlib puts ticks wherever it is asked and never checks whether the
    labels fit. With a hundred panels each axis is about 20 pt tall, and six
    powers of two on it overlap by several points - which is what made the
    1/2/4/8/16 labels run together. Every tick mark is kept, as many labels as
    there is room for are shown, and the topmost is always one of them so the
    ceiling of the scale stays readable.

    Call this after tight_layout and subplots_adjust, since only then is the
    axis its final height.
    """
    ticks = ax.get_yticks()
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
    keep = set(range(len(ticks) - 1, -1, -step))     # count down from the top
    ax.set_yticklabels(
        [lab if i in keep else '' for i, lab in enumerate(labels)],
        fontsize=fontsize)


def _rects_path(rects):
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

X_MAX_DEFAULT = 100000

# dpi savefig uses, so min_draw_width can work out how many pixels an axis gets.
# 300 for raster output, which is publication-grade and what a reader will
# zoom into. Vector output ignores it. min_draw_width reads it, so the
# one-pixel floor halves when the dpi doubles - a small region is drawn more
# faithfully at 300 than at 150. --dpi overrides it.
SAVE_DPI = 300

# Set from the output filename in __main__; see resolve_output.
VECTOR_OUTPUT = False
VECTOR_EXTS = ('.svg', '.svgz', '.pdf', '.eps', '.ps')
DEFAULT_EXT = '.svg'


def resolve_output(path):
    """
    (filename, is_vector), adding DEFAULT_EXT when no extension was given.

    SVG by default, matching make_area_graph.py. Vector output stores every
    stretch at its true width, so the track can be zoomed into and measured;
    a stretch narrower than a pixel is then faint or invisible until zoomed,
    which is accepted as the honest rendering. Raster output cannot show such
    a stretch at all - a 370 bp bar on a 5 Mb axis paints literally no pixels -
    so it gets the one-pixel floor in min_draw_width instead.
    """
    stem, ext = os.path.splitext(path)
    if not ext:
        ext = DEFAULT_EXT
        path = stem + ext
    return path, ext.lower() in VECTOR_EXTS


def min_draw_width(ax, x_max):
    """Width in data units of one pixel of this axis, or 0 for vector output."""
    if VECTOR_OUTPUT:
        return 0.0
    fig = ax.get_figure()
    axis_px = fig.get_size_inches()[0] * SAVE_DPI * ax.get_position().width
    return x_max / max(1.0, axis_px)

# Height of the strip drawn under every region, in log2 units. It marks a
# called region whose depth has fallen to 1 or 0, which a bar of height
# log2(1) = 0 could not.
BASELINE_H = 0.18


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


def read_depth(path, use_known):
    """
    Group depth rows by query.

    Returns (tracks, plasmid_regions, q_sizes, use_known) where tracks maps
    (host_id, plasmid_id) -> list of (start, end, depth, region_index).
    """
    df = pd.read_csv(path, sep='\t', comment='#')
    df.columns = df.columns.str.strip()

    required = {'Q name', 'Q start', 'Q end', 'Depth'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: missing columns in {path}: {sorted(missing)}")
        print("This script reads a *_depth.tsv, not a *_summary.tsv.")
        sys.exit(1)

    if use_known is None:
        names = df['Q name'].astype(str).unique()
        matching = sum(1 for n in names if is_known_name(n))
        use_known = len(names) > 0 and matching == len(names)
        style = 'plasmid_id,start,end,host_id' if use_known else 'plain'
        print(f"Q name format detected: {style} "
              f"({matching}/{len(names)} match the plasmid convention). "
              f"Override with --known or --no-known.")

    has_qsize = 'Q size' in df.columns
    has_region = 'Region Index' in df.columns
    tracks, plasmid_regions, q_sizes = {}, {}, {}

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
        tracks[key].append((
            int(row['Q start']), int(row['Q end']), int(row['Depth']),
            int(row['Region Index']) if has_region else 0,
        ))

    for key in tracks:
        tracks[key].sort()
    return tracks, plasmid_regions, q_sizes, use_known


def draw_panel(ax, rows, label, plasmid_region, x_max, y_max, linear, cmap, norm):
    x_step = max(1, x_max // 10)

    if plasmid_region is not None:
        p_start, p_end = plasmid_region
        ax.add_patch(patches.Rectangle(
            (p_start, -BASELINE_H), p_end - p_start, y_max + BASELINE_H,
            linewidth=0, facecolor='tab:blue', alpha=0.13, zorder=0))

    # One strip per region, so a called region is visible even where its depth
    # is 1 and its bar has no height. All the same colour, so one path.
    min_w = min_draw_width(ax, x_max)
    by_region = {}
    for start, end, _depth, region_i in rows:
        lo, hi = by_region.get(region_i, (start, end))
        by_region[region_i] = (min(lo, start), max(hi, end))
    strips = [(lo, -BASELINE_H, lo + max(hi - lo, min_w), 0.0)
              for lo, hi in by_region.values()]
    if strips:
        ax.add_patch(PathPatch(_rects_path(strips), linewidth=0,
                               edgecolor='none', facecolor='0.55', zorder=1))

    # An outline only helps while the bars are wide enough to have an inside;
    # below that it is all edge and the track turns into a picket fence.
    narrow = x_max / 250.0
    # Height and colour are both fixed by depth, so every bar at a given depth
    # is identical in style and they can share one path. That leaves a handful
    # of elements instead of one per stretch.
    groups = defaultdict(list)
    for start, end, depth, _region_i in rows:
        if depth < 1:
            continue                     # nothing to draw; the strip shows it
        height = depth if linear else math.log2(depth)
        if height <= 0:
            continue                     # depth 1, likewise
        wide = (end - start) > narrow
        groups[(depth, wide)].append(
            (start, 0.0, start + max(end - start, min_w), height))
    for (depth, wide), rects in sorted(groups.items()):
        ax.add_patch(PathPatch(
            _rects_path(rects),
            linewidth=0.4 if wide else 0,
            edgecolor='white' if wide else 'none',
            facecolor=cmap(norm(depth)), zorder=2))

    ax.set_ylabel(label, rotation=0, labelpad=22, va='center', fontsize=9)
    ax.set_xticks(range(0, x_max + x_step, x_step))
    ax.axhline(0, color='0.3', linewidth=0.6, zorder=3)

    if linear:
        step = max(1, int(y_max) // 4)
        ax.set_yticks([t for t in range(0, int(y_max) + 1) if t % step == 0])
    else:
        # Ticks at powers of two, labelled with the clade count itself. Kept
        # inside y_max because set_yticks widens the view to fit its ticks.
        ticks = [k for k in range(0, int(math.floor(y_max)) + 1)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(2 ** k) for k in ticks], fontsize=7)
    ax.tick_params(axis='x', labelsize=7)

    # Last, so neither the patches nor the ticks can widen it.
    ax.set_xlim(0, x_max)
    ax.set_ylim(-BASELINE_H, y_max)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 make_depth_graph.py <path_to_depth.tsv> <output.svg> "
              "[--fasta sequences.fasta] [--only N,N,...] [--min-depth N] "
              "[--linear] [--known | --no-known] [--no-compact] "
              "[--svg-precision N] [--dpi N]")
        sys.exit(1)

    file_path = sys.argv[1]
    # SVG by default; an explicit extension is honoured.
    output_file, VECTOR_OUTPUT = resolve_output(sys.argv[2])
    file_name = os.path.basename(file_path)

    known = None
    if '--known' in sys.argv:
        known = True
    if '--no-known' in sys.argv:
        known = False
    linear = '--linear' in sys.argv
    compact = '--no-compact' not in sys.argv

    if '--dpi' in sys.argv:
        idx = sys.argv.index('--dpi')
        if idx + 1 >= len(sys.argv):
            print("Error: --dpi requires an integer, e.g. --dpi 150")
            sys.exit(1)
        try:
            SAVE_DPI = int(sys.argv[idx + 1])
        except ValueError:
            print("Error: --dpi must be an integer.")
            sys.exit(1)

    svg_precision = None
    if '--svg-precision' in sys.argv:
        idx = sys.argv.index('--svg-precision')
        if idx + 1 >= len(sys.argv):
            print("Error: --svg-precision requires an integer, e.g. 4")
            sys.exit(1)
        try:
            svg_precision = int(sys.argv[idx + 1])
        except ValueError:
            print("Error: --svg-precision must be an integer.")
            sys.exit(1)

    fasta_lengths = {}
    if '--fasta' in sys.argv:
        idx = sys.argv.index('--fasta')
        if idx + 1 >= len(sys.argv):
            print("Error: --fasta requires a path to a FASTA file")
            sys.exit(1)
        fasta_lengths = parse_fasta_lengths(sys.argv[idx + 1])
        print(f"Loaded lengths for {len(fasta_lengths)} sequences from FASTA.")

    only_nums = set()
    if '--only' in sys.argv:
        idx = sys.argv.index('--only')
        if idx + 1 >= len(sys.argv):
            print("Error: --only requires a comma-separated list, e.g. --only 3,7,12")
            sys.exit(1)
        try:
            only_nums = set(int(x) for x in sys.argv[idx + 1].split(','))
        except ValueError:
            print("Error: --only values must be integers.")
            sys.exit(1)

    min_depth = 0
    if '--min-depth' in sys.argv:
        idx = sys.argv.index('--min-depth')
        if idx + 1 >= len(sys.argv):
            print("Error: --min-depth requires an integer.")
            sys.exit(1)
        try:
            min_depth = int(sys.argv[idx + 1])
        except ValueError:
            print("Error: --min-depth must be an integer.")
            sys.exit(1)

    if VECTOR_OUTPUT:
        print(f"Writing {output_file}: vector, so stretches are drawn at true "
              f"width with no one-pixel floor. Narrow ones are faint until "
              f"you zoom in.")
    else:
        print(f"Writing {output_file}: raster, so a stretch narrower than a "
              f"pixel is widened to one pixel to stay visible. Use an .svg "
              f"name for exact widths.")

    tracks, plasmid_regions, q_sizes, known = read_depth(file_path, known)
    tracks = dict(sorted(tracks.items()))
    all_keys = list(tracks.keys())

    if min_depth > 0:
        # Keep a query only if some stretch of it reaches min_depth. Its rows
        # are untouched, so the panels still tile their regions.
        tracks = {k: v for k, v in tracks.items()
                  if any(d >= min_depth for _s, _e, d, _r in v)}
        print(f"{len(tracks)} of {len(all_keys)} query/queries reach "
              f"depth >= {min_depth}")

    if only_nums:
        invalid = only_nums - set(range(1, len(all_keys) + 1))
        if invalid:
            print(f"Warning: out of range (max {len(all_keys)}): {sorted(invalid)}")
        keep = [all_keys[i - 1] for i in sorted(only_nums)
                if 1 <= i <= len(all_keys)]
        tracks = {k: tracks[k] for k in keep if k in tracks}

    if not tracks:
        print("Nothing to plot.")
        sys.exit(1)

    max_depth = max((d for rows in tracks.values() for _s, _e, d, _r in rows),
                    default=2)
    y_max = (max_depth * 1.08) if linear else (math.log2(max(2, max_depth)) + 0.35)

    cmap = plt.get_cmap('YlOrRd')
    # Log, not linear. Depth doubles rather than climbs, so a linear ramp
    # crushes the interesting range: against a file whose deepest stretch is
    # 16, depths 2 and 4 both land in the palest fifth of the colours and are
    # indistinguishable. LogNorm from vmin 1 is exactly a log2 normalisation -
    # the base cancels in log(d)/log(vmax) - so it matches the bar heights and
    # the y-axis ticks. Colour stays log under --linear too, where only the bar
    # heights become raw counts, since crushing the low depths is no more
    # readable there.
    norm = LogNorm(vmin=1, vmax=max(2, max_depth))

    n = len(tracks)
    print(f"Plotting {n} panel(s); deepest stretch has {max_depth} clade(s).")
    height_per_row = 1.35 if n <= 8 else max(0.55, 55 / n)
    fig = plt.figure(figsize=(14, height_per_row * n + 1.0))

    panel_axes = []
    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        original = all_keys.index(key) + 1
        ax = fig.add_subplot(n, 1, pos)
        panel_axes.append(ax)
        x_max = q_sizes.get(key) or fasta_lengths.get(key[0], X_MAX_DEFAULT)
        p_region = plasmid_regions.get(key) if known else None
        draw_panel(ax, rows, str(original), p_region, x_max, y_max,
                   linear, cmap, norm)
        ax.set_title(f'{key[0]} {key[1]}'.strip(), fontsize=9)

    axis_label = ('Clades covering' if linear
                  else 'Clades covering (log$_2$ scale)')
    fig.supylabel(axis_label, fontsize=10)
    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Clade depth - {file_name}', fontsize=11)

    plt.tight_layout(rect=[0.02, 0.01, 0.94, 0.98])
    plt.subplots_adjust(hspace=0.85)

    # The ramp is normalised to this file's deepest stretch, so say what it
    # means here rather than leaving two figures to be compared by eye.
    ticks = [2 ** k for k in range(0, int(math.floor(math.log2(
        max(2, max_depth)))) + 1)]
    if ticks[-1] != max_depth:
        ticks.append(max_depth)
    cax = fig.add_axes([0.955, 0.12, 0.011, 0.74])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       ticks=ticks)
    bar.ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    bar.ax.minorticks_off()
    bar.set_label('Clades', fontsize=8)
    bar.outline.set_linewidth(0.4)

    # Last, once every axis has its final height: drop the tick labels that
    # would not fit. Includes the colourbar, which gets short on small figures.
    for a in panel_axes + [bar.ax]:
        fit_yticks(a)

    plt.savefig(output_file,
                dpi=SAVE_DPI if VECTOR_OUTPUT
                else raster_dpi(fig, SAVE_DPI))

    # Shrink the path data before reporting, so the size printed is the size
    # on disk. Only .svg is plain text; .svgz is gzipped and .pdf/.eps are not
    # SVG at all.
    if compact and output_file.lower().endswith('.svg'):
        before, after, n = compact_svg(output_file, svg_precision)
        if n:
            print(f"Compacted {n:,} rectangle(s): {before/1e6:.2f} MB -> "
                  f"{after/1e6:.2f} MB ({100*(1-after/before):.0f}% smaller)"
                  + (f", coordinates rounded to {svg_precision} dp"
                     if svg_precision is not None else ", geometry unchanged"))
    print(f"Wrote {output_file}")
    plt.close(fig)
