#!/usr/bin/env python3
# Draws the *_depth.tsv written by alasight as a clade-breadth track: one panel
# per query, bar height and colour set by how small a fraction of the available
# clades a stretch hit.
#
# make_depth_graph.py plots how MANY clades sit on a stretch. This plots how
# FEW of the ones that could have been there actually were:
#
#     fraction = Tree Leaves Hit / Tree Leaves In LCA
#
# the numerator being the distinct tree leaves hit, the denominator the leaves
# the database can reach anywhere under their LCA. A gene inherited vertically
# turns up in most of its clade, so the fraction is near 1. A gene that arrived
# by transfer turns up in a scattered few of a large clade, so the fraction is
# small. Small is the interesting case, so small is drawn tall and dark.
#
#   python3 make_fraction_graph.py <path_to_depth.tsv> <output.svg>
#   python3 make_fraction_graph.py depth.tsv out.svg --max-fraction 0.01
#   python3 make_fraction_graph.py depth.tsv out.png --linear
#
# Flags mirror make_depth_graph.py where they overlap.

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
    from matplotlib.colors import Normalize
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

    Panels stack, so a hundred-query run is a very tall figure. Merely large
    gets a warning and proceeds; over 65,536 px in either direction matplotlib
    fails outright, so that is capped rather than left to crash. Vector output
    has neither limit.
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
    labels fit; with many panels each axis is only tens of points tall. Every
    tick mark is kept, as many labels as there is room for are shown, and the
    topmost is always one of them. Call after tight_layout.
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
    keep = set(range(len(ticks) - 1, -1, -step))
    ax.set_yticklabels(
        [lab if i in keep else '' for i, lab in enumerate(labels)],
        fontsize=fontsize)


def _rects_path(rects):
    """
    One compound Path covering many rectangles, given (x0, y0, x1, y1) each.

    The SVG backend writes one <path> element per artist, so a track of a
    hundred thousand stretches drawn as a hundred thousand Rectangles becomes a
    hundred thousand elements, each repeating its own style and clip-path.
    Collapsing rectangles that share a colour into one Path keeps every
    coordinate exactly as it was while emitting one element for the lot.
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
SAVE_DPI = 300

VECTOR_OUTPUT = False
VECTOR_EXTS = ('.svg', '.svgz', '.pdf', '.eps', '.ps')
DEFAULT_EXT = '.svg'

# Height of the strip drawn under every region, in axis units. It marks a
# called region whose bar has no height - one whose fraction is 1, or which had
# no tree-matched clades to take a fraction of at all.
BASELINE_H = 0.18

# The fraction is continuous, so unlike the depth graph there is no natural set
# of distinct colours to group rectangles by. Quantising the COLOUR into this
# many steps restores the grouping - bar heights stay exact, only the shade is
# rounded, and a sequential ramp does not resolve more steps than this by eye.
COLOR_BINS = 32


def resolve_output(path):
    """
    (filename, is_vector), adding DEFAULT_EXT when no extension was given.

    SVG by default, matching the other plotting scripts. Vector output stores
    every stretch at its true width, so the track can be zoomed into and
    measured; a stretch narrower than a pixel is then faint until zoomed, which
    is the honest rendering. Raster output cannot show such a stretch at all,
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


def height_of(fraction, linear):
    """
    Bar height for a fraction, or 0 when there is nothing to draw.

    Log by default: -log10(fraction), so a fraction of 1 is flat, 1/100 stands
    2 units tall and 1/10,000 stands 4. The interesting stretches are the small
    fractions, and this is what makes them the tall ones. --linear plots
    1 - fraction instead, which keeps the same direction on a bounded axis.
    """
    if fraction is None or fraction <= 0:
        return 0.0
    fraction = min(1.0, fraction)
    return (1.0 - fraction) if linear else -math.log10(fraction)


def read_depth(path, use_known):
    """
    Group depth rows by query, keeping the fraction for each stretch.

    Returns (tracks, plasmid_regions, q_sizes, use_known) where tracks maps
    (host_id, plasmid_id) -> list of (start, end, fraction_or_None, region_index).
    """
    df = pd.read_csv(path, sep='\t', comment='#')
    df.columns = df.columns.str.strip()

    required = {'Q name', 'Q start', 'Q end',
                'Tree Leaves Hit', 'Tree Leaves In LCA'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: missing columns in {path}: {sorted(missing)}")
        print("This script needs a depth or summary TSV from a version of "
              "alasight that writes the tree-leaf columns.")
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

        hit = int(row['Tree Leaves Hit'])
        avail = int(row['Tree Leaves In LCA'])
        if hit <= 0 or avail <= 0:
            # No tree-matched clades, so no fraction exists. Drawn as the bare
            # region strip rather than silently as a fraction of nothing.
            fraction = None
            n_no_leaves += 1
        else:
            fraction = hit / avail
            if hit > avail:
                # Only reachable when a hit leaf is in the database index but
                # absent from the tree the run used, so it counts in the
                # numerator and not the denominator.
                n_over += 1
                fraction = 1.0
        tracks[key].append((
            int(row['Q start']), int(row['Q end']), fraction,
            int(row['Region Index']) if has_region else 0,
        ))

    if n_no_leaves:
        print(f"{n_no_leaves:,} row(s) have no tree-matched clades; drawn as a "
              f"bare region strip with no bar.")
    if n_over:
        print(f"Warning: {n_over:,} row(s) hit more leaves than their LCA "
              f"contains, so a hit leaf is missing from the tree. Clamped to 1.")

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

    min_w = min_draw_width(ax, x_max)
    by_region = {}
    for start, end, _frac, region_i in rows:
        lo, hi = by_region.get(region_i, (start, end))
        by_region[region_i] = (min(lo, start), max(hi, end))
    strips = [(lo, -BASELINE_H, lo + max(hi - lo, min_w), 0.0)
              for lo, hi in by_region.values()]
    if strips:
        ax.add_patch(PathPatch(_rects_path(strips), linewidth=0,
                               edgecolor='none', facecolor='0.55', zorder=1))

    narrow = x_max / 250.0
    # Grouped by quantised colour so the whole track is a handful of elements.
    groups = defaultdict(list)
    for start, end, frac, _region_i in rows:
        height = height_of(frac, linear)
        if height <= 0:
            continue                     # fraction 1, or no fraction at all
        shade = round(norm(height) * (COLOR_BINS - 1))
        wide = (end - start) > narrow
        groups[(int(shade), wide)].append(
            (start, 0.0, start + max(end - start, min_w), height))
    for (shade, wide), rects in sorted(groups.items()):
        ax.add_patch(PathPatch(
            _rects_path(rects),
            linewidth=0.4 if wide else 0,
            edgecolor='white' if wide else 'none',
            facecolor=cmap(shade / (COLOR_BINS - 1)), zorder=2))

    ax.set_ylabel(label, rotation=0, labelpad=26, va='center', fontsize=9)
    ax.set_xticks(range(0, x_max + x_step, x_step))
    ax.axhline(0, color='0.3', linewidth=0.6, zorder=3)

    if linear:
        ticks = [t / 4 for t in range(0, 5) if t / 4 <= y_max]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{(1 - t) * 100:.0f}%" for t in ticks], fontsize=7)
    else:
        # Ticks at powers of ten, labelled with the fraction they stand for.
        ticks = list(range(0, int(math.floor(y_max)) + 1))
        ax.set_yticks(ticks)
        ax.set_yticklabels([_pct(10.0 ** -k) for k in ticks], fontsize=7)
    ax.tick_params(axis='x', labelsize=7)

    ax.set_xlim(0, x_max)
    ax.set_ylim(-BASELINE_H, y_max)


def _pct(fraction):
    """A fraction as a short percentage string, two significant figures."""
    pct = fraction * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    if pct >= 0.001:
        return f"{pct:.2g}%"
    return f"{pct:.0e}%".replace("e-0", "e-")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 make_fraction_graph.py <path_to_depth.tsv> "
              "<output.svg> [--fasta sequences.fasta] [--only N,N,...] "
              "[--max-fraction F] [--linear] [--known | --no-known] "
              "[--no-compact] [--svg-precision N] [--dpi N]")
        sys.exit(1)

    file_path = sys.argv[1]
    file_name = os.path.basename(file_path)

    output_file, VECTOR_OUTPUT = resolve_output(sys.argv[2])
    if VECTOR_OUTPUT:
        print(f"Writing {output_file}: vector, so stretches are drawn at true "
              f"width with no one-pixel floor. Narrow ones are faint until "
              f"you zoom in.")
    else:
        print(f"Writing {output_file}: raster, so a stretch narrower than a "
              f"pixel is widened to one pixel to stay visible. Use an .svg "
              f"name for exact widths.")

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

    max_fraction = None
    if '--max-fraction' in sys.argv:
        idx = sys.argv.index('--max-fraction')
        if idx + 1 >= len(sys.argv):
            print("Error: --max-fraction requires a number, e.g. 0.01")
            sys.exit(1)
        try:
            max_fraction = float(sys.argv[idx + 1])
        except ValueError:
            print("Error: --max-fraction must be a number.")
            sys.exit(1)

    tracks, plasmid_regions, q_sizes, known = read_depth(file_path, known)
    tracks = dict(sorted(tracks.items()))
    all_keys = list(tracks.keys())

    if max_fraction is not None:
        # Keep a query only if some stretch of it is at or below the cutoff.
        # Its rows are untouched, so the panels still tile their regions.
        tracks = {k: v for k, v in tracks.items()
                  if any(f is not None and f <= max_fraction
                         for _s, _e, f, _r in v)}
        print(f"{len(tracks)} of {len(all_keys)} query/queries reach a "
              f"fraction <= {max_fraction:g}")

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

    fracs = [f for rows in tracks.values() for _s, _e, f, _r in rows
             if f is not None and f > 0]
    if not fracs:
        print("No row has both a hit leaf and an LCA leaf count, so there is "
              "no fraction to draw.")
        sys.exit(1)
    smallest = min(fracs)
    heights = [height_of(f, linear) for f in fracs]
    y_max = (1.02 if linear else max(heights) + 0.35)

    # Dark for a small fraction, which is the interesting end. A different ramp
    # from make_depth_graph.py on purpose: the two figures encode different
    # quantities and should not be mistaken for one another.
    cmap = plt.get_cmap('YlGnBu')
    norm = Normalize(vmin=0.0, vmax=y_max)

    n = len(tracks)
    print(f"Plotting {n} panel(s); smallest fraction {smallest:.2e} "
          f"({_pct(smallest)} of its LCA subtree).")
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

    fig.supylabel('Fraction of the LCA subtree hit'
                  + ('' if linear else ' (log scale)'), fontsize=10)
    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Clade breadth - {file_name}', fontsize=11)

    plt.tight_layout(rect=[0.02, 0.01, 0.925, 0.98])
    plt.subplots_adjust(hspace=0.85)

    bar_ticks = ([0.0, 0.25, 0.5, 0.75, 1.0] if linear
                 else [k for k in range(0, int(math.floor(y_max)) + 1)])
    cax = fig.add_axes([0.940, 0.12, 0.011, 0.74])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       ticks=bar_ticks)
    bar.ax.set_yticklabels(
        [_pct(1 - t) if linear else _pct(10.0 ** -t) for t in bar_ticks],
        fontsize=7)
    bar.ax.minorticks_off()
    bar.set_label('Fraction hit', fontsize=8)
    bar.outline.set_linewidth(0.4)

    for a in panel_axes + [bar.ax]:
        fit_yticks(a)

    plt.savefig(output_file,
                dpi=SAVE_DPI if VECTOR_OUTPUT
                else raster_dpi(fig, SAVE_DPI))

    if compact and output_file.lower().endswith('.svg'):
        before, after, nrect = compact_svg(output_file, svg_precision)
        if nrect:
            print(f"Compacted {nrect:,} rectangle(s): {before/1e6:.2f} MB -> "
                  f"{after/1e6:.2f} MB ({100*(1-after/before):.0f}% smaller)"
                  + (f", coordinates rounded to {svg_precision} dp"
                     if svg_precision is not None else ", geometry unchanged"))
    print(f"Wrote {output_file}")
    plt.close(fig)
