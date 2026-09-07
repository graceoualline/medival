#!/usr/bin/env python3
"""
ALASIGHT - find horizontally transferred regions in a query genome.

A query FASTA is aligned against a reference database with alamem. Hits to the
query's own close relatives are dropped, then overlapping hits are paired and
kept only where the two reference species are separated by enough divergence
time on the Time Tree of Life.

Three subcommands:

  build-tree preprocess a TimeTree newick into the Euler tour and range-minimum
             tables the divergence lookups need. Optional: build-db does this
             itself when -tr is not already prepared. Useful on its own when the
             machine that has ete3 is not the one building databases.
  build-db   one-off construction of a ALASIGHT database. Resumable: every step
             is skipped if its output already exists. It runs no aligner and no
             skani: all ANI is computed per query, over just the references that
             query actually hits.
  run        the query pipeline. Not resumable - it is fast, just rerun it.

Requires alamem and skani 0.3.0+ (0.3.2+ recommended) on PATH. ete3 is needed
only to preprocess a newick; once a tree directory exists, build-db and run read
it directly and need no ete3.

Reference species come from --species-file at build time and drive the pairing
step. Query species are optional output metadata; hits to the query's own close
relatives are removed by a skani ANI check, not by taxonomy.

`run` writes five files into the output directory:

  <name>_alamem_results.tsv          raw alamem hits
  <name>_first_div_output.tsv        hits kept by the ANI filter
  <name>_overlap_div.tsv             overlapping hit pairs from different clades
  <name>_final_regions.tsv           merged / size-filtered / clustered regions
  <name>_final_regions_summary.tsv   one row per final region

Examples:
  alasight.py build-db -i genomes.txt -d gtdb_db -tr timetree/ \\
             -n "TimeTree v5 Final.nwk" --species-file sp.tsv
  alasight.py build-tree -n "TimeTree v5 Final.nwk" -o timetree/   # separately, if preferred
  alasight.py run -q genome.fa.gz -o results -d gtdb_db -t 32

Wherever a FASTA is expected you may instead pass a .txt file containing one
FASTA path per line, and any of those files may be gzipped.
"""

import argparse
import contextlib
import csv
import gzip
import heapq
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

# Files that live inside a built database directory.
DB_FASTA_LIST = "db_fasta_list.txt"
DB_SEQ_LENGTHS = "seq_lengths.tsv"
DB_INDEX = "alasight_db_index.tsv"
DB_CONFIG = "alasight_db.json"

# All ANI is computed with `skani triangle -i`. Pinned to a recent release so
# behaviour is predictable; 0.3.2 fixed -i failing outright on non-x86.
MIN_SKANI = (0, 3, 0)
MIN_SKANI_STR = "0.3.0"

# What a preprocessed tree directory must contain.
TREE_SUFFIXES = (".mins.npy", ".index.npy", ".tour_dist.npy", ".node_dists.tsv")


# Column layout shared by the divergence filter and everything downstream.
# "Divergence Time" and "ANI<95(if div=unk)" are gone: without a query species
# the first is never computed and the second is True for every surviving row.
# The divergence that still matters is between the two references, and that is
# "Div bt Ref Species" in the overlap output below.
FIRST_DIV_HEADER = [
    "Q name", "Q size", "Q start", "Q end",
    "T name", "T size", "T start", "T end",
    "Percent Identity", "Query Species", "Reference Species",
]
OVERLAP_HEADER = FIRST_DIV_HEADER + ["Div bt Ref Species", "ANI<95 bt Ref Seqs"]

# Positions within a FIRST_DIV_HEADER row, used by the overlap pairing.
_QS = 2       # Q start
_QE = 3       # Q end
_TNAME = 4    # T name
_RSP = 10     # Reference Species

# Reference-side columns aggregated across merged rows; order defines output order.
META_COLS = [
    "T name", "T size", "T start", "T end", "Percent Identity",
    "Reference Species", "Div bt Ref Species", "ANI<95 bt Ref Seqs",
]
FIXED_COLS = ["Q name", "Q size", "Q start", "Q end", "Query Species"]
SUMMARY_COLS = FIXED_COLS + ["Num Regions", "Num Unique Species", "Avg Divergence Time"]

AlamemHit = namedtuple(
    "AlamemHit",
    "t_name q_name t_size t_start t_end q_size q_start q_end strand ani score",
)


# ===========================================================================
# Small shared helpers
# ===========================================================================

@contextlib.contextmanager
def atomic(final):
    """
    Yield a scratch path to write to, and move it into place only on success.

    Every resume check in build-db is "does the output exist", so a step killed
    part-way through - disk full, OOM, walltime - must not leave a truncated file
    that the next run mistakes for a finished one. On failure the .partial is left
    behind and the real name never appears, so the step simply reruns.
    """
    final = Path(final)
    partial = final.with_name(final.name + ".partial")
    if partial.is_dir():
        shutil.rmtree(partial)
    elif partial.exists():
        partial.unlink()
    yield partial
    partial.replace(final)


class Progress:
    """
    Minimal stderr progress reporter, no dependency. On a TTY it rewrites a single
    line; when stderr is redirected - a cluster log, say - it prints one line per
    10% instead, so the log stays readable rather than filling with control codes.
    """

    def __init__(self, total, label):
        self.total = max(int(total), 1)
        self.label = label
        self.done = 0
        self.start = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.last_emit = 0.0
        self.last_decile = 0   # 0% is redundant with the line that precedes the bar

    def update(self, n=1):
        self.done += n
        frac = min(self.done / self.total, 1.0)
        now = time.monotonic()
        if self.tty:
            if self.done < self.total and now - self.last_emit < 0.2:
                return
            self.last_emit = now
            filled = int(30 * frac)
            eta = (now - self.start) * (1 - frac) / frac if frac > 0 else 0.0
            sys.stderr.write(f"\r  {self.label} [{'=' * filled}{' ' * (30 - filled)}] "
                             f"{self.done:,}/{self.total:,}  eta {eta:4.0f}s ")
            sys.stderr.flush()
        else:
            decile = int(frac * 10)
            if decile > self.last_decile:
                self.last_decile = decile
                print(f"  {self.label}: {decile * 10}% ({self.done:,}/{self.total:,})",
                      flush=True)

    def close(self):
        elapsed = time.monotonic() - self.start
        if self.tty:
            sys.stderr.write(f"\r  {self.label} [{'=' * 30}] {self.done:,}/{self.total:,}"
                             f"  done in {elapsed:.1f}s\n")
            sys.stderr.flush()
        else:
            print(f"  {self.label}: done, {self.done:,} in {elapsed:.1f}s", flush=True)


def run_cmd(command):
    """Run a subprocess, letting its stderr through so failures are visible."""
    subprocess.run([str(a) for a in command], check=True)


def resolve_fasta_inputs(path):
    """
    A '.txt' path is a newline-delimited list of FASTA paths; anything else is a
    single FASTA. Either may be gzipped. This matches how alamem and skani treat
    their own inputs, so the same argument works everywhere.
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        paths = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    else:
        paths = [path]
    return [p.resolve() for p in paths]


def open_fasta(path):
    """Open a FASTA for reading, transparently decompressing '.gz'."""
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def iter_fasta_headers(paths):
    """
    Yield (sequence_id, length) for every record across a list of FASTA files.
    alamem and skani read the FASTAs themselves, so this is the only thing the
    pipeline needs from them and a full parse would just be overhead.
    """
    for path in paths:
        with open_fasta(path) as handle:
            seq_id, length = None, 0
            for line in handle:
                if line.startswith(">"):
                    if seq_id is not None:
                        yield seq_id, length
                    seq_id, length = line[1:].split(None, 1)[0], 0
                else:
                    length += len(line.strip())
            if seq_id is not None:
                yield seq_id, length


def write_tsv(path, header, rows, config_header=""):
    with open(path, "w", newline="") as f:
        if config_header:
            f.write(config_header)
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {len(rows)} row(s) to {path}")


def config_header_text(args):
    """A '#' comment block recording how the run was invoked."""
    lines = ["# ALASIGHT run configuration",
             f"# date: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    lines += [f"# {k}: {v}" for k, v in sorted(vars(args).items())]
    return "\n".join(lines) + "\n#\n"


def check_cache(a, b, cache):
    """Symmetric two-key cache lookup."""
    if (a, b) in cache:
        return cache[(a, b)]
    if (b, a) in cache:
        return cache[(b, a)]
    return None


def overlaps(start1, end1, start2, end2):
    """Size of the overlap between two closed intervals, or None."""
    if max(start1, start2) <= min(end1, end2):
        return min(end1, end2) - max(start1, start2)
    return None


def crude_ani(len1, len2, matches):
    """ANI estimate used when a sequence is too short for skani."""
    if 0 in (len1, len2):
        return -1
    return (matches / min(len1, len2)) * 100


def crude_ani_overlap(s1, e1, s2, e2, len1, len2):
    overlap = overlaps(s1, e1, s2, e2)
    if overlap is None or 0 in (len1, len2):
        return -1
    return (overlap / min(len1, len2)) * 100


# ===========================================================================
# Divergence tree
# ===========================================================================

def _one(paths, what, tree_dir):
    paths = list(paths)
    if len(paths) == 1:
        return str(paths[0])
    found = sorted(p.name for p in paths)
    hint = ""
    if not found and any(Path(tree_dir).glob("*.tour.npy")):
        hint = ("\nThis directory holds *.tour.npy and a *.pkl, so it came from the old "
                "preprocessing script. Regenerate it with build-tree.")
    elif len(found) > 1:
        hint = "\nBuild into an empty directory so only one of each file is present."
    raise SystemExit(
        f"Expected exactly one {what} file in {tree_dir}, found {len(found)}: {found}{hint}\n"
        f"  alasight.py build-tree -n <timetree.nwk> -o <new empty directory>")


class DivergenceTree:
    """
    Time Tree of Life plus its Euler-tour preprocessing, for constant-time LCA
    lookups. Build the directory with the build-tree subcommand.
    """

    def __init__(self, tree_dir):
        d = Path(tree_dir)
        self.index = np.load(_one(d.glob("*.index.npy"), ".index.npy", d))
        self.mins = np.load(_one(d.glob("*.mins.npy"), ".mins.npy", d))
        self.tour_dist = np.load(_one(d.glob("*.tour_dist.npy"), ".tour_dist.npy", d))
        # node_dists.tsv is written in Euler tour order, and first occurrences in
        # that order are exactly preorder - which the substring fallback in
        # leaf_name relies on. No newick is read here; only build-tree needs ete3.
        self.node_dist = {}
        self.node_names = []
        with open(_one(d.glob("*.node_dists.tsv"), ".node_dists.tsv", d)) as f:
            for line in f:
                name, dist, step = line.rstrip("\n").split("\t")
                self.node_dist[name] = (float(dist), int(step))
                self.node_names.append(name)

    def _min_index(self, i, j):
        """Range minimum query over the Euler tour depths L[i:j]."""
        m = int(np.log2(j - i))
        minimum = min(self.mins[m, i], self.mins[m, j - 2 ** m])
        return self.index[m, i] if self.mins[m, i] == minimum else self.index[m, j - 2 ** m]

    def divergence(self, a, b):
        """Divergence time in MYA between two named tree nodes."""
        if a == "NA" or b == "NA":
            return "unk:unable_to_find_ref_species_in_tree"
        a_rec = self.node_dist.get(a)
        b_rec = self.node_dist.get(b)
        if a_rec is None or b_rec is None:
            return "unk:unable_to_find_ref_species_in_tree"
        a_dist, a_step = a_rec
        b_dist, b_step = b_rec
        lca_step = self._min_index(min(a_step, b_step), max(a_step, b_step) + 1)
        return (a_dist + b_dist - 2 * float(self.tour_dist[lca_step])) / 2

    def leaf_name(self, species):
        """
        Map a species name onto the name of its node in the tree, or 'NA'.

        Matched against the named nodes loaded from node_dists.tsv rather than by
        searching an ete3 tree, so the three candidate forms are O(1) dict
        lookups instead of walks over every node. Unnamed internal nodes are
        absent from that file, but a non-empty genus can never be a substring of
        an empty name, so omitting them changes no answer.
        """
        if species == "unclassified":
            return "NA"
        name = "'" + "_".join(species.split(" ")) + "'"
        parts = name.split("_")
        # Try the full name, then genus+species, then genus alone.
        for candidate in (name, parts[0] + "_" + parts[1] + "'" if len(parts) > 1 else None,
                          parts[0] + "'"):
            if candidate is not None and candidate in self.node_dist:
                return candidate
        # Last resort: first node whose name contains the genus, in preorder. The
        # leading quote is deliberate - newick labels here are quoted.
        for node_name in self.node_names:
            if parts[0] in node_name:
                return node_name
        return "NA"


# ===========================================================================
# Database index, and reference-vs-reference ANI computed per run
# ===========================================================================

def load_index(path):
    """seq_id -> (species, length, tree_leaf_name, file_index)"""
    index = {}
    four_col = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                index[parts[0]] = (parts[1], int(parts[2]), parts[3], int(parts[4]))
            elif len(parts) == 4:
                four_col += 1
    if four_col and not index:
        sys.exit(f"{path} has the old four-column layout. The fifth column says which "
                 f"reference file each sequence lives in, which the per-run triangle "
                 f"needs. Delete it and {DB_SEQ_LENGTHS}, then rerun build-db.")
    print(f"Loaded database index: {len(index):,} sequences")
    return index


def ref_species(index, seq_id):
    rec = index.get(seq_id)
    return rec[0] if rec else "unclassified"


def ref_length(index, seq_id):
    rec = index.get(seq_id)
    return rec[1] if rec else 0


def ref_leaf(index, seq_id):
    rec = index.get(seq_id)
    return rec[2] if rec else "NA"


def ref_file_idx(index, seq_id):
    """Which entry of db_fasta_list.txt holds this sequence, or None."""
    rec = index.get(seq_id)
    return rec[3] if rec else None


def _extract_hits(job):
    """Worker: copy just the wanted records out of one reference file."""
    path, wanted, out_path = job
    written = 0
    with open_fasta(path) as handle, open(out_path, "w") as out:
        keep = False
        for line in handle:
            if line.startswith(">"):
                keep = line[1:].split(None, 1)[0] in wanted
                if keep:
                    written += 1
            if keep:
                out.write(line)
    return out_path if written else None


def run_skani_ani(db_dir, query_paths, query_files, hit_names, index, threads, out_dir):
    """
    One skani triangle over the query plus the genome files holding every
    reference alamem hit. It answers both ANI questions the pipeline asks:

      query_hits[q_name] -> reference names at >= 95% ANI to that query sequence,
                            which filter 1 uses to drop hits to the query's own kin
      ref_pairs          -> (min, max) reference name pairs at >= 95% ANI,
                            which filter 2 uses to tell near-identical references apart

    Only the hit sequences are compared, but they are written out one file per
    source genome rather than pooled into a single FASTA. That grouping is load
    bearing: pooling the same sequences into one file shifts skani's ANI by up to
    ~1 here and moves a few percent of pairs across the 95% cutoff, whereas one
    file per genome is byte-identical to comparing the whole genome files and
    roughly 7x faster when a genome contributes one hit out of ten contigs.
    """
    db_files = [l.strip() for l in
                (Path(db_dir) / DB_FASTA_LIST).read_text().splitlines() if l.strip()]
    by_file = defaultdict(set)
    for name in hit_names:
        i = ref_file_idx(index, name)
        if i is not None and 0 <= i < len(db_files):
            by_file[i].add(name)

    scratch = Path(out_dir) / "skani_subset"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        jobs = [(db_files[i], names, str(scratch / f"ref_{i}.fa"))
                for i, names in sorted(by_file.items())]
        print(f"Extracting {len(hit_names):,} hit reference(s) from "
              f"{len(jobs):,} genome file(s)...")
        bar = Progress(len(jobs), "extracting references")
        subset = []
        if threads > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(threads, len(jobs))) as pool:
                for result in pool.map(_extract_hits, jobs, chunksize=8):
                    if result:
                        subset.append(result)
                    bar.update()
        else:
            for job in jobs:
                result = _extract_hits(job)
                if result:
                    subset.append(result)
                bar.update()
        bar.close()

        list_path = Path(out_dir) / "skani_input_list.txt"
        list_path.write_text("\n".join(list(query_files) + sorted(subset)) + "\n")
        raw = Path(out_dir) / "skani_ani.tsv"
        print(f"Computing ANI over {len(hit_names):,} hit reference(s)...")
        # Screening -s at 90 rather than 95, since that's an initial filter, not the final filter
        run_cmd(["skani", "triangle", "-t", threads, "-s", "90",
                 "-l", list_path, "-i", "-E", "-o", raw])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    query_set = set(query_files)
    query_hits = defaultdict(set)
    ref_pairs = set()
    with open(raw) as f:
        f.readline()  # header
        for line in f:
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            try:
                ani = float(parts[2])
            except ValueError:
                continue
            if ani < 95.0:
                continue
            # Columns 1 and 2 are the source files, which is a safer way to tell
            # query from reference than comparing sequence names.
            a_is_query = parts[0] in query_set
            b_is_query = parts[1] in query_set
            a, b = parts[5].split()[0], parts[6].split()[0]
            if a_is_query and b_is_query:
                continue                      # query against itself
            elif a_is_query:
                query_hits[a].add(b)
            elif b_is_query:
                query_hits[b].add(a)
            elif a != b:
                ref_pairs.add((min(a, b), max(a, b)))

    print(f"  {len(query_hits):,} query sequence(s) with a >= 95% reference match; "
          f"{len(ref_pairs):,} reference pair(s) at >= 95%")
    if not query_hits:
        print("  note: no query sequence is within 95% ANI of any hit reference, so "
              "filter 1 will not drop anything as a self-hit")
    return query_hits, ref_pairs


def refs_under_95(id1, id2, pairs, index, s1, e1, s2, e2, cutoff=500):
    """
    True when two reference sequences are below 95% ANI (keep the pair),
    False when they are at or above it (discard). Short sequences are not in the
    triangle, so they fall back to a crude overlap-based estimate.
    """
    len1 = ref_length(index, id1)
    len2 = ref_length(index, id2)
    if len1 <= cutoff or len2 <= cutoff:
        ani = crude_ani_overlap(s1, e1, s2, e2, len1, len2)
        return isinstance(ani, (int, float)) and ani < 95

    return (min(id1, id2), max(id1, id2)) not in pairs  # absent means ANI < 95


# ===========================================================================
# Query species assignment
# ===========================================================================

def read_species_file(path):
    """Tab-separated seq_id -> species, with null-ish values normalised."""
    nulls = {"none", "null", "na", "n/a", "", "unknown"}
    species = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 or parts[1].strip().lower() in nulls:
                species[parts[0]] = "unclassified"
            else:
                species[parts[0]] = parts[1].strip()
    return species


def resolve_query_species(args, query_ids):
    """
    Query species are recorded in the output for reference only - no filter reads
    them. Self-hits are removed by the skani ANI check in first_divergence_filter
    instead, so an unlabelled query filters exactly the same as a labelled one.
    """
    if args.species:
        return {seq_id: args.species for seq_id in query_ids}
    if args.species_file:
        return read_species_file(args.species_file)
    return {}


# ===========================================================================
# Alignment and ANI search
# ===========================================================================

def run_alamem(db_list, query, out_path, threads, min_len, min_ani):
    """Align the whole query against the streamed database in a single pass."""
    print(f"Running alamem against {db_list}...")
    run_cmd(["alamem", db_list, query, out_path, "-k", "11",
             "-t", threads, "-l", min_len, "--min-ani", min_ani])

    hits = []
    with open(out_path) as f:
        next(f, None)  # column header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 11:
                continue
            hits.append(AlamemHit(
                t_name=p[0], q_name=p[1],
                t_size=int(p[2]), t_start=int(p[3]), t_end=int(p[4]),
                q_size=int(p[5]), q_start=int(p[6]), q_end=int(p[7]),
                strand=p[8], ani=float(p[9]), score=int(p[10]),
            ))
    print(f"  {len(hits):,} alamem hit(s)")
    return hits


# ===========================================================================
# Filter 1 - divergence between query and reference species
# ===========================================================================

def first_divergence_filter(hits, query_species, index, min_identity, skani_hits):
    """
    Drop hits to references that are essentially the same organism as the query,
    i.e. at or above 95% genome-wide ANI to it. skani supplies that ANI; short
    sequences skani will not sketch fall back to a crude estimate from the number
    of matching bases.

    This replaces the old query-species divergence test, which needed kraken2 to
    label the query and TimeTree to contain both species. The two agree except at
    the margins: a reference 3 MYA diverged but still 96% ANI used to survive and
    now does not, which for transfer detection is the better call. Divergence time
    still gates the pairing step, between the two reference species.
    """
    kept = []
    for h in hits:
        if h.ani < min_identity:
            continue

        r_len = ref_length(index, h.t_name)
        if h.q_size < 500 or r_len < 500:
            ani = crude_ani(h.q_size, r_len, h.score)
            under_95 = isinstance(ani, (int, float)) and 0 <= ani < 95
        else:
            under_95 = h.t_name not in skani_hits.get(h.q_name, ())
        if not under_95:
            continue

        kept.append([
            h.q_name, str(h.q_size), str(h.q_start), str(h.q_end),
            h.t_name, str(h.t_size), str(h.t_start), str(h.t_end),
            str(h.ani), query_species.get(h.q_name, "unclassified"),
            ref_species(index, h.t_name),
        ])

    print(f"  {len(kept):,} hit(s) passed the ANI filter")
    return kept


# ===========================================================================
# Filter 2 - compress, then pair overlapping hits from different clades
# ===========================================================================

def compress(rows):
    """Merge adjacent same-species intervals within each query sequence."""
    groups = defaultdict(set)
    for row in rows:
        groups[row[0]].add(tuple(row))

    compressed = []
    for row_set in groups.values():
        rows_sorted = sorted(row_set, key=lambda x: (int(x[_QS]), int(x[_QE]), x[_TNAME]))
        if len(rows_sorted) <= 1:
            continue

        i = 0
        s_cur, e_cur = int(rows_sorted[i][_QS]), int(rows_sorted[i][_QE])
        species_cur, row_cur = rows_sorted[i][_RSP], list(rows_sorted[i])

        # Unclassified rows are passed through untouched.
        while species_cur == "unclassified" and i < len(rows_sorted):
            s_cur, e_cur = int(rows_sorted[i][_QS]), int(rows_sorted[i][_QE])
            species_cur, row_cur = rows_sorted[i][_RSP], list(rows_sorted[i])
            compressed.append(tuple(row_cur))
            i += 1

        for row in rows_sorted[i:]:
            s2, e2, species2 = int(row[_QS]), int(row[_QE]), row[_RSP]
            if species2 == "unclassified":
                compressed.append(tuple(row))
                continue
            if species2 == species_cur and overlaps(s_cur, e_cur, s2, e2):
                e_cur = max(e_cur, e2)
            else:
                row_cur[_QE] = str(e_cur)
                compressed.append(tuple(row_cur))
                s_cur, e_cur = int(row[_QS]), int(row[_QE])
                species_cur, row_cur = row[_RSP], list(row)

        row_cur[_QE] = str(e_cur)
        compressed.append(tuple(row_cur))

    return compressed


def find_overlap_and_div_max(rows, tree, pairs, index):
    """
    Pair overlapping intervals that map to different clades, largest mutual
    overlap first so the most informative pairs are never consumed by a smaller
    one. Rows from different query sequences never interact.
    """
    div_cache = {}
    ani_cache = {}
    out_rows = set()

    groups = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)

    for qrows in groups.values():
        _pair_group(qrows, out_rows, tree, pairs, index, div_cache, ani_cache)

    result = [list(r) for r in out_rows]
    print(f"  {len(result):,} overlapping pair(s) kept")
    return result


def _pair_group(rows, out_rows, tree, pairs, index, div_cache, ani_cache):
    rows = sorted(rows, key=lambda x: (int(x[_QS]), int(x[_QE])))
    n = len(rows)
    if n <= 1:
        return

    # Sweep for geometrically overlapping pairs, applying only the cheap
    # species-string check here. Everything expensive is deferred to the pop phase.
    heap = []
    for i, row_i in enumerate(rows):
        s1, e1, sp1 = int(row_i[_QS]), int(row_i[_QE]), row_i[_RSP]
        best_ov = 0

        for j in range(i + 1, n):
            s2 = int(rows[j][_QS])
            if s2 > e1:
                break  # sorted by start, nothing further can overlap row i
            if e1 - s2 < best_ov:
                break  # ceiling on any remaining overlap is below what we have

            ov = overlaps(s1, e1, s2, int(rows[j][_QE]))
            if not ov:
                continue
            if sp1 == rows[j][_RSP] and sp1 != "unclassified":
                continue

            heapq.heappush(heap, (-ov, i, j))
            best_ov = max(best_ov, ov)
            if best_ov == e1 - s1:
                break  # full coverage, nothing better exists for row i

    # Pop in decreasing overlap order. Pairs already consumed by a larger overlap
    # are skipped before any expensive lookup happens.
    used = set()
    while heap:
        _neg_ov, i, j = heapq.heappop(heap)
        if i in used or j in used:
            continue

        row1, row2 = rows[i], rows[j]
        s1, e1 = int(row1[_QS]), int(row1[_QE])
        s2, e2 = int(row2[_QS]), int(row2[_QE])
        sp1, sp2 = row1[_RSP], row2[_RSP]

        leaf1 = ref_leaf(index, row1[_TNAME])
        leaf2 = ref_leaf(index, row2[_TNAME])
        if (leaf1 == leaf2 or sp1 == sp2) and "unclassified" not in (sp1, sp2):
            continue

        div = check_cache(leaf1, leaf2, div_cache)
        if div is None:
            div = tree.divergence(leaf1, leaf2)
            div_cache[(leaf1, leaf2)] = div

        ani = "NA"
        if isinstance(div, str):
            id1, id2 = row1[_TNAME], row2[_TNAME]
            ani = check_cache(id1, id2, ani_cache)
            if ani is None:
                ani = refs_under_95(id1, id2, pairs, index, s1, e1, s2, e2)
                ani_cache[(id1, id2)] = ani

        if not ((not isinstance(div, str) and div >= 1) or ani is True):
            continue

        out_row = []
        for h in range(len(row1)):
            if h in (0, 1, 9):          # Q name, Q size, Query Species
                out_row.append(row1[h])
            elif h == _QS:
                out_row.append(str(max(s1, s2)))
            elif h == _QE:
                out_row.append(str(min(e1, e2)))
            else:
                out_row.append(f"{row1[h]},{row2[h]}")
        out_row += [str(div), str(ani)]

        used.add(i)
        used.add(j)
        out_rows.add(tuple(out_row))


# ===========================================================================
# Filter 3 - merge intervals, drop small ones, cluster what is left
# ===========================================================================

def _copy(iv):
    return {"start": iv["start"], "end": iv["end"], "rows": list(iv["rows"])}


def region_compress(ivs):
    """Merge overlapping or directly adjacent intervals."""
    if not ivs:
        return []
    ivs = sorted(ivs, key=lambda x: x["start"])
    merged = [_copy(ivs[0])]
    for iv in ivs[1:]:
        last = merged[-1]
        if iv["start"] <= last["end"] + 1:
            last["end"] = max(last["end"], iv["end"])
            last["rows"].extend(iv["rows"])
        else:
            merged.append(_copy(iv))
    return merged


def size_filter(ivs, min_size):
    return [iv for iv in ivs if (iv["end"] - iv["start"]) >= min_size]


def cluster_gap(ivs, gap):
    """Merge intervals separated by no more than `gap` bp."""
    if not ivs or gap <= 0:
        return ivs
    ivs = sorted(ivs, key=lambda x: x["start"])
    merged = [_copy(ivs[0])]
    for iv in ivs[1:]:
        last = merged[-1]
        if iv["start"] - last["end"] <= gap:
            last["end"] = max(last["end"], iv["end"])
            last["rows"].extend(iv["rows"])
        else:
            merged.append(_copy(iv))
    return merged


def _join(rows, col):
    """'|'-join a field across contributing rows, one token per original pair."""
    return "|".join(str(r.get(col, "")) for r in rows)


def format_region_row(q_name, iv):
    first = iv["rows"][0]
    out = {
        "Q name": q_name,
        "Q size": first.get("Q size", ""),
        "Q start": iv["start"],
        "Q end": iv["end"],
        "Query Species": first.get("Query Species", ""),
    }
    for col in META_COLS:
        out[col] = _join(iv["rows"], col)
    return out


def format_summary_row(q_name, iv, index):
    first = iv["rows"][0]

    leaves = set()
    for r in iv["rows"]:
        for ref_id in str(r.get("T name", "")).split(","):
            ref_id = ref_id.strip()
            if ref_id:
                leaves.add(ref_leaf(index, ref_id))

    div_vals = []
    for r in iv["rows"]:
        for v in str(r.get("Div bt Ref Species", "")).split(","):
            try:
                div_vals.append(float(v.strip()))
            except ValueError:
                pass

    return {
        "Q name": q_name,
        "Q size": first.get("Q size", ""),
        "Q start": iv["start"],
        "Q end": iv["end"],
        "Query Species": first.get("Query Species", ""),
        "Num Regions": len(iv["rows"]),
        "Num Unique Species": len(leaves),
        "Avg Divergence Time": round(sum(div_vals) / len(div_vals), 4) if div_vals else "",
    }


def final_regions(overlap_rows, min_size, gap, index):
    """Returns (region_rows, summary_rows) as lists of dicts."""
    groups = defaultdict(list)
    for row in overlap_rows:
        record = dict(zip(OVERLAP_HEADER, row))
        groups[record["Q name"]].append({
            "start": int(record["Q start"]),
            "end": int(record["Q end"]),
            "rows": [record],
        })

    regions, summaries = [], []
    for q_name, ivs in groups.items():
        for iv in cluster_gap(size_filter(region_compress(ivs), min_size), gap):
            summary = format_summary_row(q_name, iv, index)
            if summary["Num Unique Species"] <= 1:
                continue  # a region needs at least two distinct clades
            regions.append(format_region_row(q_name, iv))
            summaries.append(summary)
    return regions, summaries


# ===========================================================================
# Pipeline
# ===========================================================================

def cmd_run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(args.database)
    name = out_dir.name
    header = config_header_text(args)

    require_skani()

    # -tr is optional: the database records the tree it was built against, and
    # the index's leaf names only mean anything relative to that tree.
    config = read_db_config(db_dir)
    tree_dir = args.tree or config.get("tree")
    if not tree_dir:
        sys.exit(f"No tree directory. Pass -tr, or rebuild {db_dir} so it records one.")
    if args.tree and config.get("tree") and Path(args.tree).resolve() != Path(config["tree"]):
        print(f"  WARNING: -tr {args.tree} is not the tree this database was built against "
              f"({config['tree']}). The index's leaf names came from that tree.")

    print("Start time:", datetime.now())
    print(f"Query: {args.query}\nDatabase: {db_dir}\nTree: {tree_dir}\n"
          f"Threads: {args.threads}\n")

    tree = DivergenceTree(tree_dir)
    index = load_index(db_dir / DB_INDEX)

    query_paths = resolve_fasta_inputs(args.query)
    query_ids = [seq_id for seq_id, _ in iter_fasta_headers(query_paths)]
    print(f"Read {len(query_ids)} query sequence(s) from {len(query_paths)} file(s)")
    query_species = resolve_query_species(args, query_ids)

    hits = run_alamem(db_dir / DB_FASTA_LIST, args.query,
                      out_dir / f"{name}_alamem_results.tsv",
                      args.threads, args.min_len, args.min_ani)

    # After alamem, so only the references it actually hit are compared.
    query_hits, ref_pairs = run_skani_ani(
        db_dir, query_paths, [str(p) for p in query_paths],
        {h.t_name for h in hits}, index, args.threads, out_dir)

    print("Filtering out references too similar to the query...")
    first_div = first_divergence_filter(hits, query_species, index,
                                        args.min_ani, query_hits)
    write_tsv(out_dir / f"{name}_first_div_output.tsv", FIRST_DIV_HEADER, first_div, header)

    print("Pairing overlapping hits...")
    overlap = find_overlap_and_div_max(compress(first_div), tree, ref_pairs, index)
    write_tsv(out_dir / f"{name}_overlap_div.tsv", OVERLAP_HEADER, overlap, header)

    print("Building final regions...")
    regions, summaries = final_regions(overlap, args.size_filter, args.cluster_size, index)
    region_cols = FIXED_COLS + META_COLS
    write_tsv(out_dir / f"{name}_final_regions.tsv", region_cols,
              [[r[c] for c in region_cols] for r in regions], header)
    write_tsv(out_dir / f"{name}_final_regions_summary.tsv", SUMMARY_COLS,
              [[s[c] for c in SUMMARY_COLS] for s in summaries], header)

    print("ALASIGHT FINISHED")
    print("End time:", datetime.now())


# ===========================================================================
# Tree preprocessing (resumable)
# ===========================================================================

def euler_tour(root):
    """
    Depth-first Euler tour: every node is recorded again each time the walk
    returns to it, so consecutive depths differ by exactly one and the lowest
    common ancestor of two nodes is the shallowest entry between their positions.

    Returns (levels, dists, node_dist), where node_dist maps each named node to
    its (distance from root, last position in the tour). Any occurrence of a node
    works for an LCA query, so keeping only the last one is enough.
    """
    levels, dists = [0], [0.0]
    node_dist = {}
    if root.name:
        node_dist[root.name] = (0.0, 0)

    stack = [(root, 0, 0.0, iter(root.children))]
    while stack:
        node, depth, dist, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            if stack:  # stepping back up records the parent again
                parent, p_depth, p_dist, _ = stack[-1]
                levels.append(p_depth)
                dists.append(p_dist)
                if parent.name:
                    node_dist[parent.name] = (p_dist, len(levels) - 1)
        else:
            c_depth, c_dist = depth + 1, dist + child.dist
            stack.append((child, c_depth, c_dist, iter(child.children)))
            levels.append(c_depth)
            dists.append(c_dist)
            if child.name:
                node_dist[child.name] = (c_dist, len(levels) - 1)
    return levels, dists, node_dist


def build_sparse_table(levels):
    """
    Sparse table for range-minimum queries: mins[m, n] is the smallest depth in
    levels[n : n + 2**m] and index[m, n] is where it occurs. A block running off
    the end keeps its left half, and ties take the right, which is what the
    query in _min_index expects.
    """
    depths = np.asarray(levels, dtype=np.int64)
    n = depths.size
    rows = int(np.log2(n)) + 1
    mins = np.zeros((rows, n), dtype=np.int64)
    index = np.zeros((rows, n), dtype=np.int64)
    mins[0] = depths
    index[0] = np.arange(n, dtype=np.int64)

    sentinel = np.iinfo(np.int64).max
    for m in range(1, rows):
        half = 1 << (m - 1)
        right_min = np.full(n, sentinel, dtype=np.int64)
        right_idx = np.zeros(n, dtype=np.int64)
        if half < n:
            right_min[:n - half] = mins[m - 1, half:]
            right_idx[:n - half] = index[m - 1, half:]
        take_left = mins[m - 1] < right_min
        mins[m] = np.where(take_left, mins[m - 1], right_min)
        index[m] = np.where(take_left, index[m - 1], right_idx)
    return mins, index


def tree_dir_ready(tree_dir):
    """True when a directory already holds a full set of preprocessed tree files."""
    d = Path(tree_dir)
    return all(any(d.glob(f"*{suf}")) for suf in TREE_SUFFIXES)


def build_tree(newick, output):
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(newick).stem

    if all((out_dir / f"{stem}{suf}").exists() for suf in TREE_SUFFIXES):
        print(f"Skipping tree preprocessing, outputs already exist in {out_dir}")
        return

    nwk_copy = out_dir / f"{stem}.nwk"
    # Two newicks in one directory would break the *.nwk glob at load time.
    others = [p.name for p in out_dir.glob("*.nwk") if p.name != nwk_copy.name]
    if others:
        sys.exit(f"{out_dir} already contains a different newick: {others}. "
                 f"Build into an empty directory instead.")
    if not nwk_copy.exists():
        shutil.copy(newick, nwk_copy)

    # Imported here, not at module scope, and only reached when a tree actually
    # needs preprocessing - so run, and build-db against a prepared tree
    # directory, need no ete3 installed at all.
    from ete3 import Tree

    print(f"Parsing {newick}...")
    tree = Tree(str(nwk_copy))

    print("Walking the Euler tour...")
    levels, dists, node_dist = euler_tour(tree)
    print(f"  {len(levels):,} tour steps, {len(node_dist):,} named node(s)")

    print("Building the range-minimum sparse table...")
    mins, index = build_sparse_table(levels)
    print(f"  table is {mins.shape[0]} x {mins.shape[1]}")

    for suffix, array in ((".mins.npy", mins), (".index.npy", index),
                          (".tour_dist.npy", np.asarray(dists, dtype=np.float64))):
        # np.save appends .npy to a path that lacks it, so hand it a file object.
        with atomic(out_dir / f"{stem}{suffix}") as tmp, open(tmp, "wb") as fh:
            np.save(fh, array)
    with atomic(out_dir / f"{stem}.node_dists.tsv") as tmp, open(tmp, "w") as f:
        for name, (dist, step) in node_dist.items():
            f.write(f"{name}\t{dist}\t{step}\n")

    print(f"Tree preprocessing complete -> {out_dir}")


def cmd_build_tree(args):
    print("Start time:", datetime.now())
    build_tree(args.newick, args.output)
    print("Pass this directory as -tr to build-db and run.")
    print("End time:", datetime.now())


# ===========================================================================
# Database construction (resumable)
# ===========================================================================

def _scan_one_fasta(job):
    """Worker for build_seq_lengths; module level so it can be pickled."""
    file_idx, path = job
    return file_idx, list(iter_fasta_headers([path]))


def build_seq_lengths(inputs, out_path, threads):
    if out_path.exists():
        print(f"Skipping sequence lengths, {out_path} exists")
        return
    print(f"Writing sequence lengths from {len(inputs)} file(s) on {threads} worker(s)...")
    # Third column is the position in db_fasta_list.txt; the per-run triangle uses
    # it to pass whole genome files rather than extracting individual sequences.
    #
    # gzip decompression is the one CPU-bound Python step in build-db and the GIL
    # makes threads useless for it, so files are split across processes. map()
    # preserves input order, so the output is identical to the serial version.
    jobs = list(enumerate(inputs))
    bar = Progress(len(jobs), "scanning FASTAs")
    with atomic(out_path) as tmp, open(tmp, "w") as f:
        if threads > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(threads, len(jobs))) as pool:
                for file_idx, records in pool.map(_scan_one_fasta, jobs, chunksize=8):
                    for seq_id, length in records:
                        f.write(f"{seq_id}\t{length}\t{file_idx}\n")
                    bar.update()
        else:
            for job in jobs:
                file_idx, records = _scan_one_fasta(job)
                for seq_id, length in records:
                    f.write(f"{seq_id}\t{length}\t{file_idx}\n")
                bar.update()
    bar.close()


def build_fasta_list(inputs, out_path):
    """
    One absolute FASTA path per line. alamem streams this file directly and skani
    triangle reads the same file with -l, so the database never copies the
    reference sequences and gzipped inputs stay gzipped.
    """
    if out_path.exists():
        print(f"Skipping FASTA list, {out_path} exists")
        return
    with atomic(out_path) as tmp:
        tmp.write_text("\n".join(str(p) for p in inputs) + "\n")
    print(f"Wrote database FASTA list ({len(inputs)} file(s)) to {out_path}")


def skani_version():
    """(major, minor, patch) from `skani -V`, or None if it cannot be parsed."""
    try:
        out = subprocess.run(["skani", "-V"], capture_output=True, text=True,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(g) for g in match.groups()) if match else None


def require_skani():
    """
    ANI is computed with `triangle -i`, one entry per sequence. Pinning to a
    recent release keeps behaviour predictable; 0.3.2 in particular fixed -i
    failing outright on non-x86 platforms.
    """
    version = skani_version()
    if version is None:
        sys.exit(f"Could not determine the skani version; {MIN_SKANI_STR} or newer is required.")
    if version < MIN_SKANI:
        sys.exit(f"skani {'.'.join(map(str, version))} is too old - {MIN_SKANI_STR} or newer "
                 f"is required. 0.3.2+ is recommended: earlier 0.3.x releases had a logic "
                 f"error where -i failed outright on non-x86 platforms.")
    if version < (0, 3, 2):
        print(f"  note: skani {'.'.join(map(str, version))} detected; 0.3.2 fixed -i failing "
              f"on non-x86 platforms")
    return version


def write_db_config(db_dir, args, n_reference_files):
    """
    Record what this database was built from, so `run` does not need the same
    paths passed again. The tree matters most: the leaf names in the index came
    from that specific tree directory, and pairing against a different one would
    silently mismatch.
    """
    existing = read_db_config(db_dir)
    existing.update({
        "created": datetime.now().isoformat(timespec="seconds"),
        "tree": str(Path(args.tree).resolve()),
        "species_file": str(Path(args.species_file).resolve()),
        "reference_input": str(Path(args.input).resolve()),
        "n_reference_files": n_reference_files,
    })
    with atomic(Path(db_dir) / DB_CONFIG) as tmp:
        tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"  recorded build settings in {Path(db_dir) / DB_CONFIG}")


def read_db_config(db_dir):
    """Build settings recorded by build-db, or {} if absent or unreadable."""
    path = Path(db_dir) / DB_CONFIG
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        print(f"  WARNING: could not parse {path}; ignoring it")
        return {}


def build_index(db_dir, species_file, tree_dir):
    """seq_id -> species, length, tree leaf name, as a TSV."""
    out_path = db_dir / DB_INDEX
    if out_path.exists():
        print(f"Skipping database index, {out_path} exists")
        return

    # Loaded only once we know the work is needed - parsing the TimeTree newick
    # is not free, and on a resumed build it would be for nothing.
    species_map = read_species_file(species_file)
    tree = DivergenceTree(tree_dir)

    print("Building database index...")
    leaf_cache = {}
    total = covered = unusable = 0
    with atomic(out_path) as tmp, open(db_dir / DB_SEQ_LENGTHS) as lengths, open(tmp, "w") as out:
        for line in lengths:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            seq_id, length = parts[0], parts[1]
            file_idx = parts[2] if len(parts) > 2 else "0"
            species = species_map.get(seq_id, "unclassified")
            if species not in leaf_cache:
                leaf_cache[species] = tree.leaf_name(species)
            out.write(f"{seq_id}\t{species}\t{length}\t{leaf_cache[species]}\t{file_idx}\n")
            total += 1
            if seq_id in species_map:
                covered += 1
            if leaf_cache[species] == "NA":
                unusable += 1

    print(f"  index written to {out_path}")
    print(f"  {total:,} sequences | {covered:,} found in the species file "
          f"({100.0 * covered / total if total else 0:.1f}%) | "
          f"{total - unusable:,} with a usable tree leaf "
          f"({100.0 * (total - unusable) / total if total else 0:.1f}%)")
    print(f"  {len(leaf_cache):,} distinct species, "
          f"{sum(1 for v in leaf_cache.values() if v == 'NA'):,} of which are not in the tree")
    if total and covered / total < 0.5:
        print("  WARNING: over half the references are missing from the species file. "
              "Check that its first column holds sequence IDs, not assembly accessions.")
    if total and (total - unusable) / total < 0.5:
        print("  WARNING: over half the references have no tree leaf and cannot contribute "
              "to the pairing step, so most regions will be dropped.")


def cmd_build_db(args):
    db_dir = Path(args.database)
    db_dir.mkdir(parents=True, exist_ok=True)

    require_skani()
    print("Start time:", datetime.now())

    # The tree is an input to the index step, so prepare it first if needed.
    # Pointing several databases at one tree directory just resumes.
    if not tree_dir_ready(args.tree):
        if not args.newick:
            sys.exit(f"{args.tree} holds no preprocessed tree. Pass -n/--newick "
                     f"<timetree.nwk> to build it there, or build it separately with:\n"
                     f"  alasight.py build-tree -n <timetree.nwk> -o {args.tree}")
        build_tree(args.newick, args.tree)

    inputs = resolve_fasta_inputs(args.input)
    print(f"Reference input: {len(inputs)} FASTA file(s)")

    build_seq_lengths(inputs, db_dir / DB_SEQ_LENGTHS, args.threads)
    build_fasta_list(inputs, db_dir / DB_FASTA_LIST)
    build_index(db_dir, args.species_file, args.tree)

    write_db_config(db_dir, args, len(inputs))

    print("DATABASE BUILD COMPLETE")
    print(f"Run queries against it with:  alasight.py run -q <query> -o <out> -d {db_dir}")
    print("End time:", datetime.now())


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the query pipeline")
    run.add_argument("-q", "--query", required=True,
                     help="query FASTA, or a .txt file listing FASTA paths; .gz is fine")
    run.add_argument("-o", "--output", required=True, help="output directory")
    run.add_argument("-d", "--database", required=True, help="ALASIGHT database directory")
    run.add_argument("-tr", "--tree",
                     help="preprocessed Time Tree directory; defaults to the one the "
                          "database was built against")
    run.add_argument("-s", "--species-file",
                     help="TSV of query sequence ID and species (output metadata only)")
    run.add_argument("--species",
                     help="label every query sequence with this species (output metadata only; "
                          "no filter uses it)")
    run.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1)
    run.add_argument("--min-len", type=int, default=40,
                     help="minimum alamem hit length in bp (default: 40)")
    run.add_argument("--min-ani", type=float, default=90.0,
                     help="minimum percent identity / ANI of a hit (default: 90)")
    run.add_argument("--size-filter", type=int, default=150,
                     help="drop final regions smaller than this many bp (default: 150)")
    run.add_argument("--cluster-size", type=int, default=0,
                     help="merge final regions within this many bp (default: 0)")
    run.set_defaults(func=cmd_run)

    build = sub.add_parser("build-db", help="build a ALASIGHT database")
    build.add_argument("-i", "--input", required=True,
                       help="reference multi-FASTA, or a .txt file listing FASTA paths; .gz is fine")
    build.add_argument("-d", "--database", required=True, help="database directory to create")
    build.add_argument("-tr", "--tree", required=True,
                       help="Time Tree directory; preprocessed here if it is not already")
    build.add_argument("-n", "--newick",
                       help="TimeTree newick, used only if -tr is not already preprocessed")
    build.add_argument("-s", "--species-file", required=True,
                       help="TSV of reference sequence ID and species; the divergence pairing "
                            "depends on it, so it is required")
    build.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1)
    build.set_defaults(func=cmd_build_db)

    tree_cmd = sub.add_parser("build-tree",
                              help="preprocess a TimeTree newick for fast LCA lookups")
    tree_cmd.add_argument("-n", "--newick", required=True, help="TimeTree newick file")
    tree_cmd.add_argument("-o", "--output", required=True,
                          help="tree directory to create; pass it as -tr afterwards")
    tree_cmd.set_defaults(func=cmd_build_tree)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
