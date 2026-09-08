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

import sys
import os

import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
except ModuleNotFoundError:
    print("Error: Matplotlib is not installed.")
    print("Please install Matplotlib by running: pip install matplotlib")
    sys.exit(1)
import math

X_MAX_DEFAULT = 100000

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
    # is 1 and its bar has no height.
    by_region = {}
    for start, end, _depth, region_i in rows:
        lo, hi = by_region.get(region_i, (start, end))
        by_region[region_i] = (min(lo, start), max(hi, end))
    for lo, hi in by_region.values():
        ax.add_patch(patches.Rectangle(
            (lo, -BASELINE_H), hi - lo, BASELINE_H,
            linewidth=0, facecolor='0.55', zorder=1))

    # An outline only helps while the bars are wide enough to have an inside;
    # below that it is all edge and the track turns into a picket fence.
    narrow = x_max / 250.0
    for start, end, depth, _region_i in rows:
        if depth < 1:
            continue                     # nothing to draw; the strip shows it
        height = depth if linear else math.log2(depth)
        if height <= 0:
            continue                     # depth 1, likewise
        wide = (end - start) > narrow
        ax.add_patch(patches.Rectangle(
            (start, 0), end - start, height,
            linewidth=0.4 if wide else 0, edgecolor='white' if wide else 'none',
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
        print("Usage: python3 make_depth_graph.py <path_to_depth.tsv> <output.png> "
              "[--fasta sequences.fasta] [--only N,N,...] [--min-depth N] "
              "[--linear] [--known | --no-known]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_file = sys.argv[2]
    file_name = os.path.basename(file_path)

    known = None
    if '--known' in sys.argv:
        known = True
    if '--no-known' in sys.argv:
        known = False
    linear = '--linear' in sys.argv

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

    for pos, (key, rows) in enumerate(tracks.items(), start=1):
        original = all_keys.index(key) + 1
        ax = fig.add_subplot(n, 1, pos)
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

    plt.savefig(output_file, dpi=300)
    print(f"Wrote {output_file}")
    plt.close(fig)
