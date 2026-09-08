#!/usr/bin/env python3
# This code will take in the output .txt file of alasight
# and output a graph that shows all regions of detected
# mobile elements in a genome
# must have matplotlib installed

import sys
import re
import pandas as pd
import numpy as np
import csv
import os
import subprocess
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
except ModuleNotFoundError:
    print("Error: Matplotlib is not installed.")
    print("Please install Matplotlib by running: pip install matplotlib")
    exit()
import math

# None means "decide from the data" (see get_coor);
# --known / --no-known override it.
# When on, plots a rectangle showing where the
# known plasmid is, and requires "Q name" to have the format:
# plasmid_id,plasmid_start,plasmid_end,host_id
known = None

# Fallback x-axis range used when no FASTA is provided or sequence ID is not found
X_MAX_DEFAULT = 100000

# dpi both savefig calls use, so min_draw_width below can work out how many
# pixels an axis really gets. 300 for raster output, which is publication-grade
# and what a reader will zoom into; vector output ignores it. min_draw_width
# reads it, so the one-pixel floor halves when the dpi doubles - a small region
# is drawn more faithfully at 300 than at 150. --dpi overrides it.
SAVE_DPI = 300

# Set from the output filename in __main__. A raster needs the one-pixel floor
# in min_draw_width or sub-pixel regions vanish, but a vector file has no pixel
# grid to accommodate: baking the floor into its geometry would widen every
# small region permanently, and zooming in would show a rectangle several times
# the size of the region it stands for.
VECTOR_OUTPUT = False
VECTOR_EXTS = ('.svg', '.svgz', '.pdf', '.eps', '.ps')

# SVG unless the output name says otherwise. Vector output stores every region
# at its true width, so a 200 bp region measures 200 bp however far out the
# view is, and the figure can be zoomed and measured rather than only glanced
# at. The cost is accepted deliberately: on a whole chromosome a small region
# is a sub-pixel sliver and will look faint or invisible until zoomed. That is
# the honest rendering. A raster cannot do it - it has to widen such a region
# to a whole pixel to show it at all - so .png still gets the floor.
DEFAULT_EXT = '.svg'


def resolve_output(path):
    """(filename, is_vector), adding DEFAULT_EXT when no extension was given."""
    stem, ext = os.path.splitext(path)
    if not ext:
        ext = DEFAULT_EXT
        path = stem + ext
    return path, ext.lower() in VECTOR_EXTS


def min_draw_width(ax, x_max):
    """
    Width in data units of one pixel of this axis, or 0 for vector output.

    A rectangle narrower than a pixel cannot be drawn honestly in a raster. On
    a 5 Mb chromosome an axis holds roughly 2,600 bp per pixel, so a 200 bp
    region is 0.08 px wide: drawn plainly it antialiases away to nothing, and
    drawn with the 1 pt edge this script used to give it, it paints about 4 px
    - a 50x overstatement, which is what turned a genome with a few small
    regions into a solid red band. Widening it to exactly one pixel keeps it
    visible while overstating it as little as a raster allows.

    Vector output gets no floor. There the rectangle is stored at its true
    width and the reader can zoom until it is visible, which is the point of
    asking for SVG in the first place.
    """
    if VECTOR_OUTPUT:
        return 0.0
    fig = ax.get_figure()
    axis_px = (fig.get_size_inches()[0] * SAVE_DPI * ax.get_position().width)
    return x_max / max(1.0, axis_px)


# Definition of what a "known" Q name is
# The middle two fields must be integers.
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


def raster_dpi(fig, dpi, warn_mpx=60):
    """
    The dpi to actually use, given how large the raster would be.

    Panels stack, so a many-genome run can be a very tall figure. Merely large
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


def _rects_path(rects):
    """
    One compound Path covering many rectangles, given (x0, y0, x1, y1) each.

    The SVG backend writes one <path> element per artist, so a genome with
    thousands of regions drawn as thousands of Rectangles becomes thousands of
    elements, each repeating its own style and clip-path attributes. Collapsing
    rectangles that share a colour into a single Path keeps every coordinate
    exactly as it was while emitting one element for the lot.

    The regions reaching here never overlap - compress() guarantees it - so
    filling them in one path paints the same pixels as filling them one by one.
    Where the one-pixel floor widens two neighbours into contact, one path is
    in fact the better rendering: a single alpha 0.5 fill rather than two
    stacked into a darker spot that means nothing.
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


def compact_svg(path, precision=None):
    """
    Rewrite the rectangle subpaths matplotlib emitted into a shorter form.

    Matplotlib writes every rectangle as `M x0 y0 L x1 y0 L x1 y1 L x0 y1 z`
    with a newline after each command - eight numbers where four will do. The
    same corners as `M x0 y0 H x1 V y1 H x0 z` cost about 45% fewer characters,
    and no coordinate changes, so the geometry is untouched.

    `precision`, if given, also rounds coordinates to that many decimals. It is
    off by default so the output is exactly what matplotlib computed.
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


def save_figure(out_path, compact, precision):
    """Write the figure, then shrink its path data when it is plain SVG."""
    vector = os.path.splitext(out_path)[1].lower() in VECTOR_EXTS
    plt.savefig(out_path,
                dpi=SAVE_DPI if vector else raster_dpi(plt.gcf(), SAVE_DPI))
    if compact and out_path.lower().endswith('.svg'):
        before, after, n = compact_svg(out_path, precision)
        if n:
            print(f"Compacted {n:,} rectangle(s): {before/1e6:.2f} MB -> "
                  f"{after/1e6:.2f} MB ({100*(1-after/before):.0f}% smaller)"
                  + (f", coordinates rounded to {precision} dp"
                     if precision is not None else ", geometry unchanged"))
    print(f"Wrote {out_path}")


def compress(intervals):
    """Merge overlapping or adjacent intervals. Carries a 3rd element (e.g. num_species)
    by taking the max when merging; intervals without a 3rd element are also supported.

    Intervals are half-open [start, end), as alasight writes them: Q end is one
    past the last base, so a region's length is end - start.
    """
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for iv in sorted_ivs[1:]:
        s, e = iv[0], iv[1]
        # FIX: was `s <= merged[-1][1] + 1`, which is adjacency for CLOSED
        # intervals - under [a, b] the next one is contiguous at s == b + 1.
        # These are half-open, so contiguity is s == b, and the +1 was fusing
        # regions separated by one uncovered base. That matters on real output:
        # at the default --cluster-size 0 alasight leaves two regions exactly
        # one base apart precisely because that base had fewer than two clades
        # on it, and the merge below carries the label by max, so the weaker
        # region would inherit the stronger one's clade count.
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            if len(iv) > 2 and len(merged[-1]) > 2:
                merged[-1][2] = max(merged[-1][2], iv[2])
        else:
            merged.append(list(iv))
    return [tuple(iv) for iv in merged]


def size_filter(intervals, min_size):
    """Remove intervals smaller than min_size bp. Works with 2- or 3-element tuples."""
    return [iv for iv in intervals if (iv[1] - iv[0]) >= min_size]


def cluster_gap(intervals, gap):
    """Merge intervals within `gap` bp of each other. Carries 3rd element via max."""
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


def apply_filters(intervals, min_size, gap):
    """Size filter first, then cluster by gap distance."""
    result = size_filter(intervals, min_size)
    result = cluster_gap(result, gap)
    return result


def compute_score(raw_intervals, plasmid_regions, q_sizes, min_size, cluster_dist):
    """Compute detection metrics over the same intervals the figure draws.
    Applies compress, then size filter, then cluster - the order the plot uses
    (compress happens in get_coor, apply_filters runs after it).
    Uses actual entry count as denominator (interactive_filter.py hardcodes 100
    for its specific 100-plasmid test set).
    """
    arg_metrics = []
    for key in plasmid_regions:
        p_start, p_end = plasmid_regions[key]
        # NOTE: p_start/p_end come from the Q name, and are compared below
        # against alasight coordinates, which are 0-based with an exclusive
        # end. If whatever generated those names used 1-based inclusive bounds
        # (the usual annotation convention) then mge_bp is short by one and
        # every overlap is shifted by one. Worth confirming against that script.
        mge_bp = p_end - p_start
        host_bp = max(0, q_sizes.get(key, 0) - mge_bp)

        # FIX: compress first, as the plot does. Without it the printed scores
        # could contradict the figure outright - a pair of sub-threshold
        # regions that compress fuses past --size-filter is drawn as a red
        # block while the score, filtering the unfused pair, reports 0% found.
        # It also stops bases being counted twice in bp_within / bp_outside
        # when the input intervals overlap, as they do in a hits file rather
        # than a region summary.
        intervals = size_filter(compress(raw_intervals.get(key, [])), min_size)
        intervals = cluster_gap(intervals, cluster_dist)

        bp_within = bp_outside = 0
        for iv in intervals:
            s, e = iv[0], iv[1]
            overlap = max(0, min(e, p_end) - max(s, p_start))
            bp_within += overlap
            bp_outside += (e - s) - overlap

        coverage = min(1.0, bp_within / mge_bp) if mge_bp > 0 else 0
        arg_metrics.append({
            'coverage':   coverage,
            'bp_within':  bp_within,
            'bp_outside': bp_outside,
            'mge_bp':     mge_bp,
            'host_bp':    host_bp,
        })

    if not arg_metrics:
        return None

    found = [m for m in arg_metrics if m['coverage'] > 0]
    total = len(arg_metrics)
    pct_found  = len(found) / total
    total_host = sum(m['host_bp']    for m in arg_metrics)
    pct_host   = (sum(m['bp_outside'] for m in arg_metrics) / total_host) if total_host > 0 else 0
    total_mge  = sum(m['mge_bp']     for m in found)
    pct_area   = (sum(m['bp_within'] for m in found) / total_mge) if total_mge > 0 else 0
    count_90 = sum(1 for m in found if m['coverage'] > 0.9)

    return {
        'pct_found': pct_found,
        'pct_host':  pct_host,
        'pct_area':  pct_area,
        'count_90':  count_90,
    }


def graph_genome(name_tuple, coordinates, row, length, plasmid_region=None, x_max=X_MAX_DEFAULT):
    x_step = max(1, x_max // 10)
    ax = fig.add_subplot(length, 1, row)
    ax.set_title(f'{name_tuple[0]} {name_tuple[1]}')
    ax.set_ylabel(str(row), rotation=0, labelpad=20, va='center', fontsize=9)

    max_coor = 0

    # Draw known plasmid region first (background), if provided
    if plasmid_region is not None:
        p_start, p_end = plasmid_region
        rect_known = patches.Rectangle(
            (p_start, 0), p_end - p_start, 1,
            linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.2,
            label='Known plasmid region'
        )
        ax.add_patch(rect_known)
        if p_end > max_coor:
            max_coor = p_end

    # Draw detected hit regions on top (merged to avoid muddy overlap).
    # Collected first, then emitted as a single path: they all share one
    # colour, so there is no reason to spend an element on each.
    min_w = min_draw_width(ax, x_max)
    rects, labels = [], []
    for iv in coordinates:
        start1, end1 = iv[0], iv[1]
        ns = iv[2] if len(iv) > 2 else None
        # FIX: was linewidth=1, edgecolor='red'. A 1 pt edge is drawn at a
        # fixed size in points no matter how few bases the rectangle spans, so
        # every sub-pixel region painted ~4 px of solid red and a sparse genome
        # looked fully covered. No edge, and never thinner than one pixel, so
        # a region stays visible without claiming more width than it has.
        rects.append((start1, 0.0, start1 + max(end1 - start1, min_w), 1.0))
        if ns is not None:
            labels.append(((start1 + end1) / 2, ns))
        if end1 > max_coor:
            max_coor = end1
    if rects:
        ax.add_patch(PathPatch(_rects_path(rects), linewidth=0,
                               edgecolor='none', facecolor='red', alpha=0.5))
    for x_mid, ns in labels:
        ax.text(x_mid, 0.5, str(ns), ha='center', va='center',
                fontsize=7, color='darkred', fontweight='bold')

    # Hide y-axis
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xticks(range(0, x_max + x_step, x_step))
    ax.set_xlim(0, x_max)
    ax.set_xlabel('Genomic Position')



def parse_fasta_lengths(fasta_path):
    """Return {seq_id: length} for every sequence in a FASTA file."""
    lengths = {}
    current_id = None
    current_len = 0
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


def get_coor(file_path, use_known):
    # genome_dict { (host_id, plasmid_id): [compressed (Q start, Q end[, num_species]) intervals] }
    # raw_dict    { (host_id, plasmid_id): [(Q start, Q end) uncompressed, for scoring] }
    # plasmid_region_dict { (host_id, plasmid_id): (plasmid_start, plasmid_end) }
    genome_dict = {}
    raw_dict = {}
    plasmid_region_dict = {}
    q_sizes = {}

    df = pd.read_csv(file_path, sep='\t', comment='#')
    df.columns = df.columns.str.strip()

    required_cols = {'Q start', 'Q end', 'Q name'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: Missing columns in input file: {missing}")
        sys.exit(1)

    # Autodetection. The plasmid convention is only used when every Q name
    # actually follows it, so ordinary accessions fall back to plain grouping
    # instead of silently scoring 0% against a 0-0 plasmid region.
    if use_known is None:
        unique_names = df['Q name'].astype(str).unique()
        matching = sum(1 for n in unique_names if is_known_name(n))
        use_known = len(unique_names) > 0 and matching == len(unique_names)
        style = 'plasmid_id,start,end,host_id' if use_known else 'plain'
        print(f"Q name format detected: {style} "
              f"({matching}/{len(unique_names)} name(s) match the plasmid convention). "
              f"Override with --known or --no-known.")

    has_qsize = 'Q size' in df.columns
    label_col = next((c for c in ('Peak Clades', 'Num Clades', 'Num Unique Species')
                      if c in df.columns), None)
    if label_col:
        print(f"Labelling regions with '{label_col}'.")
    bad_names = set()

    for _, row in df.iterrows():
        q_name = str(row['Q name'])
        q_start = int(row['Q start'])
        q_end = int(row['Q end'])

        if use_known:
            # Expected format: plasmid_id,plasmid_start,plasmid_end,host_id
            if not is_known_name(q_name):
                bad_names.add(q_name)
                plasmid_id, plasmid_start, plasmid_end, host_id = "", 0, 0, q_name
            else:
                plasmid_id, plasmid_start, plasmid_end, host_id = q_name.split(',')
                plasmid_start = int(plasmid_start)
                plasmid_end = int(plasmid_end)
            id_tuple = (host_id, plasmid_id)
            plasmid_region_dict[id_tuple] = (plasmid_start, plasmid_end)
        else:
            # Use Q name as-is for grouping; no second label
            id_tuple = (q_name, '')

        if id_tuple not in genome_dict:
            genome_dict[id_tuple] = []
            raw_dict[id_tuple] = []
            if has_qsize:
                q_sizes[id_tuple] = int(row['Q size'])

        raw_dict[id_tuple].append((q_start, q_end))
        if label_col:
            genome_dict[id_tuple].append((q_start, q_end, int(row[label_col])))
        else:
            genome_dict[id_tuple].append((q_start, q_end))

    if bad_names:
        print(f"Warning: {len(bad_names)} Q name(s) do not match "
              f"plasmid_id,plasmid_start,plasmid_end,host_id, e.g. "
              f"{sorted(bad_names)[:3]}. Their plasmid region is recorded as 0-0, so they "
              f"will score 0% found. Use --no-known if these are ordinary names.")

    for id_tuple in genome_dict:
        print("before", len(genome_dict[id_tuple]))
        genome_dict[id_tuple] = compress(genome_dict[id_tuple])
        print("after", len(genome_dict[id_tuple]))
    return genome_dict, plasmid_region_dict, q_sizes, raw_dict, use_known


if __name__ == "__main__":
    # Usage:
    #   python3 make_area_graph_known.py <file> <output>
    #   python3 make_area_graph_known.py <file> <output> --fasta sequences.fasta
    #   python3 make_area_graph_known.py <file> <output> --all
    #   python3 make_area_graph_known.py <file> <output> --only 20,22,31
    #   python3 make_area_graph_known.py <file> <output> --size-filter 200 --cluster 500
    #   python3 make_area_graph_known.py <file> <output> --no-known
    if len(sys.argv) < 3:
        print("Usage: python3 make_area_graph_known.py <path_to_alasight.txt> <output_file> "
              "[--fasta sequences.fasta] [--all | --only N,N,...] [--size-filter N] [--cluster N] "
              "[--known | --no-known] [--no-compact] [--svg-precision N] [--dpi N]")
        sys.exit(1)

    # CHANGED: was `known = True`. Default None autodetects; the flags force it.
    known = None
    if '--known' in sys.argv:
        known = True
    if '--no-known' in sys.argv:
        known = False

    file_path = sys.argv[1]
    file_name = os.path.basename(file_path)

    # SVG by default; an explicit extension is honoured.
    output_file, VECTOR_OUTPUT = resolve_output(sys.argv[2])
    if VECTOR_OUTPUT:
        print(f"Writing {output_file}: vector, so regions are drawn at true "
              f"width with no one-pixel floor. Small ones are faint until "
              f"you zoom in.")
    else:
        print(f"Writing {output_file}: raster, so a region narrower than a "
              f"pixel is widened to one pixel to stay visible. Use an .svg "
              f"name for exact widths.")

    fasta_lengths = {}
    if '--fasta' in sys.argv:
        idx = sys.argv.index('--fasta')
        if idx + 1 >= len(sys.argv):
            print("Error: --fasta requires a path to a FASTA file")
            sys.exit(1)
        fasta_lengths = parse_fasta_lengths(sys.argv[idx + 1])
        print(f"Loaded lengths for {len(fasta_lengths)} sequences from FASTA.")

    show_all = '--all' in sys.argv
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

    only_nums = set()
    if '--only' in sys.argv:
        idx = sys.argv.index('--only')
        if idx + 1 >= len(sys.argv):
            print("Error: --only requires a comma-separated list of numbers, e.g. --only 20,22,31")
            sys.exit(1)
        try:
            only_nums = set(int(x) for x in sys.argv[idx + 1].split(','))
        except ValueError:
            print("Error: --only values must be integers, e.g. --only 20,22,31")
            sys.exit(1)

    min_size = 0
    if '--size-filter' in sys.argv:
        idx = sys.argv.index('--size-filter')
        if idx + 1 >= len(sys.argv):
            print("Error: --size-filter requires a value in bp, e.g. --size-filter 200")
            sys.exit(1)
        try:
            min_size = int(sys.argv[idx + 1])
        except ValueError:
            print("Error: --size-filter value must be an integer.")
            sys.exit(1)

    cluster_dist = 0
    if '--cluster' in sys.argv:
        idx = sys.argv.index('--cluster')
        if idx + 1 >= len(sys.argv):
            print("Error: --cluster requires a value in bp, e.g. --cluster 500")
            sys.exit(1)
        try:
            cluster_dist = int(sys.argv[idx + 1])
        except ValueError:
            print("Error: --cluster value must be an integer.")
            sys.exit(1)

    genome_positions, plasmid_regions, q_sizes, raw_intervals, known = get_coor(file_path, known)
    genome_positions = dict(sorted(genome_positions.items()))

    # Apply size filter then clustering (if requested)
    if min_size > 0 or cluster_dist > 0:
        genome_positions = {
            k: apply_filters(v, min_size, cluster_dist)
            for k, v in genome_positions.items()
        }
        print(f"Applied: size filter >= {min_size} bp, cluster gap <= {cluster_dist} bp")

    # Print detection scores (same metrics as interactive_filter.py)
    scores = compute_score(raw_intervals, plasmid_regions, q_sizes, min_size, cluster_dist)
    if scores:
        print(f"\nScores ({file_name}):")
        print(f"  % MGEs found:          {scores['pct_found']*100:6.1f}%"
              f"   |  % host genome covered:  {scores['pct_host']*100:6.2f}%")
        print(f"  % area of found MGEs:  {scores['pct_area']*100:6.1f}%"
              f"   |  MGEs >90% found:          {scores['count_90']}")
    elif not q_sizes:
        print("\nNote: 'Q size' column not found — host coverage score unavailable.")

    num_genes = len(genome_positions)
    if num_genes == 0:
        print("No data found in input file.")
        sys.exit(1)

    # Apply --only filter: keep only entries at the specified 1-based positions
    if only_nums:
        all_keys = list(genome_positions.keys())
        invalid = only_nums - set(range(1, num_genes + 1))
        if invalid:
            print(f"Warning: numbers out of range (max {num_genes}): {sorted(invalid)}")
        selected_keys = [all_keys[i - 1] for i in sorted(only_nums) if 1 <= i <= num_genes]
        genome_positions = {k: genome_positions[k] for k in selected_keys}
        num_genes = len(genome_positions)
        if num_genes == 0:
            print("No entries match the specified numbers.")
            sys.exit(1)
        show_all = True  # always show as one combined plot when filtering

    if show_all or num_genes <= 6:
        height_per_row = 1.5 if num_genes <= 6 else max(0.6, 60 / num_genes)
        fig = plt.figure(figsize=(14, height_per_row * num_genes))

        # When filtering, preserve the original row numbers in the labels
        all_original_keys = list(sorted(get_coor(file_path, known)[0].keys())) if only_nums else None  # [0] = genome_dict

        for plot_pos, gene in enumerate(genome_positions, start=1):
            original_num = (all_original_keys.index(gene) + 1) if all_original_keys else plot_pos
            p_region = plasmid_regions.get(gene) if known else None
            x_max = q_sizes.get(gene) or fasta_lengths.get(gene[0], X_MAX_DEFAULT)
            x_step = max(1, x_max // 10)
            ax = fig.add_subplot(num_genes, 1, plot_pos)
            ax.set_title(f'{gene[0]} {gene[1]}')
            ax.set_ylabel(str(original_num), rotation=0, labelpad=20, va='center', fontsize=9)

            max_coor = 0
            if p_region is not None:
                p_start, p_end = p_region
                ax.add_patch(patches.Rectangle(
                    (p_start, 0), p_end - p_start, 1,
                    linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.2,
                    label='Known plasmid region'
                ))
                max_coor = max(max_coor, p_end)
            min_w_inline = min_draw_width(ax, x_max)
            rects, labels = [], []
            for iv in genome_positions[gene]:
                start1, end1 = iv[0], iv[1]
                ns = iv[2] if len(iv) > 2 else None
                # FIX: as in graph_genome - no fixed-size edge, and never
                # thinner than one pixel. See min_draw_width.
                rects.append((start1, 0.0,
                              start1 + max(end1 - start1, min_w_inline), 1.0))
                if ns is not None:
                    labels.append(((start1 + end1) / 2, ns))
                max_coor = max(max_coor, end1)
            # One path for the lot; they all share a colour. See _rects_path.
            if rects:
                ax.add_patch(PathPatch(_rects_path(rects), linewidth=0,
                                       edgecolor='none', facecolor='red',
                                       alpha=0.5))
            for x_mid, ns in labels:
                ax.text(x_mid, 0.5, str(ns), ha='center', va='center',
                        fontsize=7, color='darkred', fontweight='bold')

            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(range(0, x_max + x_step, x_step))
            ax.set_xlim(0, x_max)
            ax.set_xlabel('Genomic Position')
            #if p_region is not None:
            #    ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.8)
        save_figure(output_file, compact, svg_precision)
        plt.show()
        plt.close(fig)
    else:
        num_rows = 6
        key_vals = list(genome_positions.items())
        num_figs = math.ceil(num_genes / num_rows)
        gene_num = 0
        for i in range(num_figs):
            fig = plt.figure(figsize=(9, 1.5 * num_rows))

            row = 0
            for gene_index in range(gene_num, min(gene_num + num_rows, num_genes)):
                gene_name = key_vals[gene_index][0]
                gene_vals = key_vals[gene_index][1]
                p_region = plasmid_regions.get(gene_name) if known else None
                x_max = q_sizes.get(gene_name) or fasta_lengths.get(gene_name[0], X_MAX_DEFAULT)
                graph_genome(gene_name, gene_vals, row + 1, num_rows, p_region, x_max)
                row = (row + 1) % num_rows

            gene_num += num_rows
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.8)
            # Keep the format the output name asked for, so a request for .svg
            # is not silently written as .png with VECTOR_OUTPUT set for it.
            stem, ext = os.path.splitext(output_file)
            save_figure(f'{stem}_plot_{i+1}{ext or ".png"}', compact,
                        svg_precision)
            plt.show()
            plt.close(fig)
