#!/usr/bin/env python3
# Draws the *_depth.tsv written by alasight as a mirrored track: one panel per
# query, clade depth above the axis and clade breadth below it.
#
#   up    log2(Depth)            how MANY clade groups sit on a stretch.
#                                Taller is more clades stacked on one locus.
#
#   down  -log10(fraction),      how FEW of the clades that could have been
#         fraction =             there actually were. A gene inherited
#         Tree Leaves Hit /      vertically turns up in most of its clade, so
#         Tree Leaves In LCA     the fraction is near 1 and the bar is flat. A
#                                gene that arrived by transfer turns up in a
#                                scattered few of a large clade, so the
#                                fraction is small and the bar is deep.
#
# Both bars come from the same row, so they share an x extent: a stretch with a
# tall bar up and a deep bar down is hit by many clades that are a small
# fraction of what was available - scattered breadth, which is the case worth
# looking at.
#
#   python3 make_mirror_graph.py <path_to_depth.tsv> <output.svg>
#   python3 make_mirror_graph.py depth.tsv out.svg --min-depth 3 --max-fraction 0.05
#
# Flags mirror make_depth_graph.py where they overlap. There is no --linear:
# the two halves measure different things, and putting a raw count above a
# bounded fraction on one axis makes a panel that cannot be read.

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
    from matplotlib.colors import LogNorm, Normalize
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

    Thinned outward from zero rather than downward from the top, because this
    axis runs both ways and dropping labels from one end only would leave it
    lopsided. Zero carries no label anyway - it is depth 1 going up and 100%
    going down at the same time - so the extremes of both halves survive.
    Call after tight_layout, when the axis has its final height.
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
    keep = {i for i, t in enumerate(ticks) if int(round(abs(t))) % step == 0}
    keep |= {0, len(ticks) - 1}                  # both extremes, always
    ax.set_yticklabels(
        [lab if i in keep else '' for i, lab in enumerate(labels)],
        fontsize=fontsize)


def _rects_path(rects):
    """
    One compound Path covering many rectangles, given (x0, y0, x1, y1) each.

    The SVG backend writes one <path> element per artist, so a track of a
    hundred thousand stretches drawn as Rectangles becomes a hundred thousand
    elements, each repeating its own style and clip-path. Collapsing rectangles
    that share a colour into one Path keeps every coordinate exactly as it was
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
SAVE_DPI = 300

VECTOR_OUTPUT = False
VECTOR_EXTS = ('.svg', '.svgz', '.pdf', '.eps', '.ps')
DEFAULT_EXT = '.svg'

# Height of the strip marking a called region, centred on zero and drawn behind
# the bars. It is what shows where a region is when neither half has a bar -
# depth 1 above, fraction 1 below.
BASELINE_H = 0.16

# The fraction is continuous, so unlike depth there is no natural set of
# distinct colours to group rectangles by. Quantising the COLOUR into this many
# steps restores the grouping; bar depths stay exact, only the shade is
# rounded.
COLOR_BINS = 32


def resolve_output(path):
    """
    (filename, is_vector), adding DEFAULT_EXT when no extension was given.

    SVG by default. Vector output stores every stretch at its true width, so
    the track can be zoomed into and measured; a stretch narrower than a pixel
    is then faint until zoomed, which is the honest rendering. Raster output
    cannot show such a stretch at all, so it gets the one-pixel floor in
    min_draw_width instead.
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


def _pct(fraction):
    """A fraction as a short percentage string, two significant figures."""
    pct = fraction * 100
    if pct >= 1:
        return f"{pct:.0f}%"
    if pct >= 0.001:
        return f"{pct:.2g}%"
    return f"{pct:.0e}%".replace("e-0", "e-")


def read_track(path, use_known):
    """
    Group rows by query, keeping both quantities for each stretch.

    Returns (tracks, plasmid_regions, q_sizes, use_known) where tracks maps
    (host_id, plasmid_id) -> list of (start, end, depth, fraction, region_index).
    `depth` is 0 where there is nothing to draw upward and `fraction` is None
    where no fraction exists.
    """
    df = pd.read_csv(path, sep='\t', comment='#')
    df.columns = df.columns.str.strip()

    # A summary TSV has no Depth column, but Peak Clades is the same quantity
    # per region, so the script reads either file.
    depth_col = next((c for c in ('Depth', 'Peak Clades', 'Num Clades')
                      if c in df.columns), None)
    required = {'Q name', 'Q start', 'Q end',
                'Tree Leaves Hit', 'Tree Leaves In LCA'}
    missing = required - set(df.columns)
    if missing or depth_col is None:
        if missing:
            print(f"Error: missing columns in {path}: {sorted(missing)}")
        if depth_col is None:
            print(f"Error: {path} has none of Depth, Peak Clades, Num Clades.")
        print("This script needs a depth or summary TSV from a version of "
              "alasight that writes the tree-leaf columns.")
        sys.exit(1)
    print(f"Reading clade counts from '{depth_col}'.")

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
            fraction = None            # no tree-matched clades, so no fraction
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
            int(row['Q start']), int(row['Q end']), int(row[depth_col]),
            fraction, int(row['Region Index']) if has_region else 0,
        ))

    if n_no_leaves:
        print(f"{n_no_leaves:,} row(s) have no tree-matched clades, so nothing "
              f"is drawn below the axis for them.")
    if n_over:
        print(f"Warning: {n_over:,} row(s) hit more leaves than their LCA "
              f"contains, so a hit leaf is missing from the tree. Clamped to 1.")

    for key in tracks:
        tracks[key].sort()
    return tracks, plasmid_regions, q_sizes, use_known


def draw_panel(ax, rows, label, plasmid_region, x_max, y_top, y_bot,
               depth_cmap, depth_norm, frac_cmap, frac_norm):
    """Depth above the axis, fraction below it, on one shared x extent."""
    x_step = max(1, x_max // 10)

    if plasmid_region is not None:
        p_start, p_end = plasmid_region
        ax.add_patch(patches.Rectangle(
            (p_start, -y_bot), p_end - p_start, y_bot + y_top,
            linewidth=0, facecolor='tab:blue', alpha=0.13, zorder=0))

    min_w = min_draw_width(ax, x_max)

    # One strip per region, centred on zero and behind everything, so a region
    # with no bar either way is still visible.
    by_region = {}
    for start, end, _d, _f, region_i in rows:
        lo, hi = by_region.get(region_i, (start, end))
        by_region[region_i] = (min(lo, start), max(hi, end))
    strips = [(lo, -BASELINE_H / 2, lo + max(hi - lo, min_w), BASELINE_H / 2)
              for lo, hi in by_region.values()]
    if strips:
        ax.add_patch(PathPatch(_rects_path(strips), linewidth=0,
                               edgecolor='none', facecolor='0.45', zorder=1))

    narrow = x_max / 250.0

    # Upward: grouped by depth, which fixes both height and colour.
    up = defaultdict(list)
    for start, end, depth, _f, _r in rows:
        if depth < 2:
            continue                   # log2(1) is flat, and 0 has no bar
        wide = (end - start) > narrow
        up[(depth, wide)].append(
            (start, 0.0, start + max(end - start, min_w), math.log2(depth)))
    for (depth, wide), rects in sorted(up.items()):
        ax.add_patch(PathPatch(
            _rects_path(rects),
            linewidth=0.4 if wide else 0,
            edgecolor='white' if wide else 'none',
            facecolor=depth_cmap(depth_norm(depth)), zorder=2))

    # Downward: grouped by quantised colour, depths kept exact.
    down = defaultdict(list)
    for start, end, _d, frac, _r in rows:
        if frac is None or frac >= 1.0:
            continue                   # nothing below the axis to draw
        drop = -math.log10(max(frac, 1e-12))
        shade = round(frac_norm(drop) * (COLOR_BINS - 1))
        wide = (end - start) > narrow
        down[(int(shade), wide)].append(
            (start, -drop, start + max(end - start, min_w), 0.0))
    for (shade, wide), rects in sorted(down.items()):
        ax.add_patch(PathPatch(
            _rects_path(rects),
            linewidth=0.4 if wide else 0,
            edgecolor='white' if wide else 'none',
            facecolor=frac_cmap(shade / (COLOR_BINS - 1)), zorder=2))

    ax.set_ylabel(label, rotation=0, labelpad=30, va='center', fontsize=9)
    ax.set_xticks(range(0, x_max + x_step, x_step))
    ax.axhline(0, color='0.25', linewidth=0.7, zorder=3)

    # Powers of two going up, powers of ten going down. Zero is left blank: it
    # is depth 1 above the line and 100% below it at once.
    ticks = list(range(-int(math.floor(y_bot)), int(math.floor(y_top)) + 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        ["" if k == 0 else (str(2 ** k) if k > 0 else _pct(10.0 ** k))
         for k in ticks], fontsize=7)
    ax.tick_params(axis='x', labelsize=7)

    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_bot, y_top)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 make_mirror_graph.py <path_to_depth.tsv> "
              "<output.svg> [--fasta sequences.fasta] [--only N,N,...] "
              "[--min-depth N] [--max-fraction F] [--known | --no-known] "
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
    compact = '--no-compact' not in sys.argv


    def _int_flag(name):
        if name not in sys.argv:
            return None
        idx = sys.argv.index(name)
        if idx + 1 >= len(sys.argv):
            print(f"Error: {name} requires an integer.")
            sys.exit(1)
        try:
            return int(sys.argv[idx + 1])
        except ValueError:
            print(f"Error: {name} must be an integer.")
            sys.exit(1)


    dpi_flag = _int_flag('--dpi')
    if dpi_flag is not None:
        SAVE_DPI = dpi_flag
    svg_precision = _int_flag('--svg-precision')
    min_depth = _int_flag('--min-depth') or 0

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
            print("Error: --only requires a comma-separated list, e.g. 3,7,12")
            sys.exit(1)
        try:
            only_nums = set(int(x) for x in sys.argv[idx + 1].split(','))
        except ValueError:
            print("Error: --only values must be integers.")
            sys.exit(1)

    tracks, plasmid_regions, q_sizes, known = read_track(file_path, known)
    tracks = dict(sorted(tracks.items()))
    all_keys = list(tracks.keys())

    # Both filters keep a whole query when any stretch of it qualifies, so the
    # panels still tile their regions.
    if min_depth > 0:
        tracks = {k: v for k, v in tracks.items()
                  if any(d >= min_depth for _s, _e, d, _f, _r in v)}
        print(f"{len(tracks)} of {len(all_keys)} query/queries reach "
              f"depth >= {min_depth}")
    if max_fraction is not None:
        before = len(tracks)
        tracks = {k: v for k, v in tracks.items()
                  if any(f is not None and f <= max_fraction
                         for _s, _e, _d, f, _r in v)}
        print(f"{len(tracks)} of {before} remaining query/queries reach a "
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

    depths = [d for rows in tracks.values() for _s, _e, d, _f, _r in rows]
    fracs = [f for rows in tracks.values() for _s, _e, _d, f, _r in rows
             if f is not None and 0 < f < 1.0]
    max_depth = max(depths) if depths else 2
    drops = [-math.log10(f) for f in fracs] or [0.0]

    # No rescaling of either half: one unit is one doubling above the line and
    # one order of magnitude below it. The halves are therefore whatever size
    # their own data makes them, which is honest - the two quantities are not
    # comparable, and the ticks carry the meaning.
    y_top = math.log2(max(2, max_depth)) + 0.35
    y_bot = max(drops) + 0.35

    depth_cmap = plt.get_cmap('YlOrRd')
    depth_norm = LogNorm(vmin=1, vmax=max(2, max_depth))
    frac_cmap = plt.get_cmap('YlGnBu')
    frac_norm = Normalize(vmin=0.0, vmax=y_bot)

    n = len(tracks)
    print(f"Plotting {n} panel(s); deepest stretch {max_depth} clade(s), "
          f"smallest fraction {_pct(min(fracs)) if fracs else 'n/a'}.")
    height_per_row = 1.7 if n <= 8 else max(0.7, 70 / n)
    fig = plt.figure(figsize=(14, height_per_row * n + 1.2))

    panel_axes = []
    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        original = all_keys.index(key) + 1
        ax = fig.add_subplot(n, 1, pos)
        panel_axes.append(ax)
        x_max = q_sizes.get(key) or fasta_lengths.get(key[0], X_MAX_DEFAULT)
        p_region = plasmid_regions.get(key) if known else None
        draw_panel(ax, rows, str(original), p_region, x_max, y_top, y_bot,
                   depth_cmap, depth_norm, frac_cmap, frac_norm)
        ax.set_title(f'{key[0]} {key[1]}'.strip(), fontsize=9)

    # Two labels rather than one: a rotated supylabel reads bottom-to-top, so a
    # single string would put the upward half's caption at the bottom of the
    # figure and the downward half's at the top - backwards from the panels.
    #
    # Each gets about 40% of the figure's height to run along, so on a
    # one-panel figure - under three inches tall - a 10 pt caption would be
    # longer than the space and the two would run into each other. Sized to
    # fit instead.
    up_label = '\u2191 clades (log$_2$)'
    dn_label = '\u2193 fraction hit (log$_{10}$)'
    budget_pt = fig.get_size_inches()[1] * 72 * 0.42
    longest = max(len(up_label), len(dn_label))
    label_pt = max(5.5, min(10.0, budget_pt / (longest * 0.58)))
    fig.text(0.012, 0.73, up_label, rotation=90, va='center', ha='center',
             fontsize=label_pt)
    fig.text(0.012, 0.27, dn_label, rotation=90, va='center', ha='center',
             fontsize=label_pt)
    fig.supxlabel('Genomic Position', fontsize=10)
    fig.suptitle(f'Clade depth and breadth - {file_name}', fontsize=11)

    plt.tight_layout(rect=[0.035, 0.01, 0.915, 0.98])
    plt.subplots_adjust(hspace=0.9)

    # Two ramps, stacked the way the panels are: clades above, fraction below.
    up_ticks = [2 ** k for k in range(0, int(math.floor(math.log2(
        max(2, max_depth)))) + 1)]
    if up_ticks[-1] != max_depth:
        up_ticks.append(max_depth)
    cax_up = fig.add_axes([0.935, 0.55, 0.011, 0.32])
    bar_up = fig.colorbar(ScalarMappable(norm=depth_norm, cmap=depth_cmap),
                          cax=cax_up, ticks=up_ticks)
    bar_up.ax.set_yticklabels([str(t) for t in up_ticks], fontsize=7)
    bar_up.ax.minorticks_off()
    bar_up.set_label('Clades', fontsize=8)
    bar_up.outline.set_linewidth(0.4)

    down_ticks = list(range(0, int(math.floor(y_bot)) + 1))
    cax_dn = fig.add_axes([0.935, 0.13, 0.011, 0.32])
    bar_dn = fig.colorbar(ScalarMappable(norm=frac_norm, cmap=frac_cmap),
                          cax=cax_dn, ticks=down_ticks)
    bar_dn.ax.set_yticklabels([_pct(10.0 ** -t) for t in down_ticks], fontsize=7)
    bar_dn.ax.minorticks_off()
    bar_dn.set_label('Fraction hit', fontsize=8)
    bar_dn.outline.set_linewidth(0.4)

    for a in panel_axes + [bar_up.ax, bar_dn.ax]:
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
