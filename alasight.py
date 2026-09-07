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
  build-db   one-off construction of an ALASIGHT database. Resumable: every step
             is skipped if its output already exists. It runs no aligner and no
             skani: all ANI is computed per query, over just the references that
             query actually hits.
  run        the query pipeline. Only the alamem alignment resumes, against the
             sidecar config it writes; every filter after it is cheap and reruns.

Requires alamem and skani 0.3.0+ (0.3.2+ recommended) on PATH, and the
pydustmasker package. ete3 is needed only to preprocess a newick; once a tree
directory exists, build-db and run read it directly and need no ete3.

Reference species come from --species-file at build time and drive the pairing
step. Query species are optional output metadata; hits to the query's own close
relatives are removed by a skani ANI check, not by taxonomy.

`run` writes eight files into the output directory:

  <name>_alamem_results.tsv            raw alamem hits
  <name>_alamem_config.json            size and mtime of the alignment inputs
  <name>_first_div_output.tsv          hits kept by the ANI filter
  <name>_overlap_div.tsv               overlapping hit pairs from different clades
  <name>_clustered_regions.tsv         merged / size-filtered / clustered regions
  <name>_clustered_regions_summary.tsv one row per clustered region
  <name>_dust_regions.tsv              regions left after DUST masking
  <name>_dust_regions_summary.tsv      one row per surviving region

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
SUMMARY_COLS = FIXED_COLS + ["Num Regions", "Num Unique Species", "Avg Divergence Time",
                             "Num Tree Pairs", "Evidence"]

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


_RANK_PREFIX = re.compile(r"^[dpcofgs]__", re.IGNORECASE)
_GTDB_SUFFIX = re.compile(r"^[A-Z]{1,2}$")
_NOISE_TOKENS = {"candidatus"}


def taxon_tokens(label):
    """
    Lowercase name tokens for a taxon label, with everything that stops a
    TimeTree match removed: newick quoting, GTDB rank prefixes (s__, g__), GTDB
    polyphyly suffixes (Escherichia_D -> escherichia) and 'Candidatus'.

    Applied to both sides of the comparison, so the two only have to agree on
    the names themselves and not on how they were decorated.
    """
    label = _RANK_PREFIX.sub("", label.strip().strip("'\""))
    tokens = []
    for token in re.split(r"[\s_;]+", label):
        if token and not _GTDB_SUFFIX.match(token) and token.lower() not in _NOISE_TOKENS:
            tokens.append(token.lower())
    return tokens


# ===========================================================================
# Divergence tree
# ===========================================================================

def _one(paths, what, tree_dir):
    paths = list(paths)
    if len(paths) == 1:
        return str(paths[0])
    found = sorted(p.name for p in paths)
    hint = ""
    if len(found) > 1:
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
        # that order are exactly preorder - which the genus fallback in leaf_match
        # relies on. No newick is read here; only build-tree needs ete3.
        self.node_dist = {}
        self.node_names = []
        with open(_one(d.glob("*.node_dists.tsv"), ".node_dists.tsv", d)) as f:
            for line in f:
                name, dist, step = line.rstrip("\n").split("\t")
                self.node_dist[name] = (float(dist), int(step))
                self.node_names.append(name)

        # Normalised name -> node name, and genus -> first node in that genus.
        # Built once here so the genus fallback is a dict lookup rather than a
        # walk over every node, and so it can be anchored at a token boundary:
        # 'Nitrospirae' is no longer a match for Nitrospira. First occurrence
        # wins, which in tour order is preorder, as before.
        self.by_norm, self.by_genus = {}, {}
        for node_name in self.node_names:
            key = "_".join(taxon_tokens(node_name))
            if not key:
                continue
            self.by_norm.setdefault(key, node_name)
            self.by_genus.setdefault(key.split("_", 1)[0], node_name)

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

    def leaf_match(self, species):
        """
        (node name or 'NA', the route that matched). Routes, in the order tried:
        'species' exact, 'genus_species' after dropping strain tokens, 'genus'
        for a named genus node, then 'genus_member' - some species of the right
        genus, when the genus has no node of its own. The first three are exact;
        'genus_member' is an approximation and worth watching in the counts
        build_index prints.

        Matched against the named nodes loaded from node_dists.tsv rather than by
        searching an ete3 tree, so every candidate is an O(1) dict lookup instead
        of a walk over every node. Unnamed internal nodes are absent from that
        file, and normalise to an empty key, so omitting them changes no answer.
        """
        if species == "unclassified":
            return "NA", "unclassified"
        tokens = taxon_tokens(species)
        if not tokens:
            return "NA", "unnamed"

        for route, key in (("species", "_".join(tokens)),
                           ("genus_species", "_".join(tokens[:2])),
                           ("genus", tokens[0])):
            hit = self.by_norm.get(key)
            if hit:
                return hit, route

        hit = self.by_genus.get(tokens[0])
        return (hit, "genus_member") if hit else ("NA", "no_match")


# ===========================================================================
# Database index, and reference-vs-reference ANI computed per run
# ===========================================================================

def load_index(path):
    """seq_id -> (species, length, tree_leaf_name, file_index)"""
    index = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                index[parts[0]] = (parts[1], int(parts[2]), parts[3], int(parts[4]))
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


def run_skani_ani(db_dir, query_files, hit_names, index, threads, out_dir,
                  representatives=False):
    """
    The ANI the pipeline needs, by whichever of two routes fits the database.

      query_hits[q_name] -> reference sequence names that are the same organism
                            as that query sequence, which filter 1 uses to drop
                            hits to the query's own kin
      ref_pairs          -> (min, max) reference name pairs at >= 95% ANI, which
                            filter 2 uses to tell near-identical references apart

    With `representatives` the database holds one genome per species - GTDB's
    species representative set, say. GTDB defines a species as a ~95% ANI cluster
    and picks representatives so no two are within 95% of each other, so every
    distinct pair of references is below the cutoff by construction. ref_pairs is
    therefore empty without computing anything, which is the answer a triangle
    would spend O(hits^2) comparisons reaching, and only the query-vs-genome half
    is actually run.
    """
    db_files = [l.strip() for l in
                (Path(db_dir) / DB_FASTA_LIST).read_text().splitlines() if l.strip()]
    by_file = defaultdict(set)
    for name in hit_names:
        i = ref_file_idx(index, name)
        if i is not None and 0 <= i < len(db_files):
            by_file[i].add(name)

    if representatives:
        return query_vs_genomes(db_files, by_file, query_files, threads, out_dir), set()
    return query_and_ref_triangle(db_files, by_file, query_files, hit_names,
                                  threads, out_dir)


def query_vs_genomes(db_files, by_file, query_files, threads, out_dir):
    """
    query_hits[q_name] -> every hit sequence belonging to a reference genome that
    is the same organism as that query sequence, i.e. >= 95% ANI to it.

    Whole reference genomes are compared, not individual hit contigs, because
    that is the granularity the question is asked at: with one genome per species
    the organism *is* the genome, and 95% ANI is where GTDB draws the species
    boundary to begin with. Three things follow. No hit sequences need
    extracting, so the per-genome scratch files and the -i grouping question both
    go away. There are far fewer entries than there are hit contigs. And a short
    contig skani cannot sketch can no longer go missing from the output and be
    read as "not a self-hit" - a whole genome always sketches.

    A self-match is expanded back to every hit sequence in that genome before
    returning, so filter 1 keeps comparing sequence names and needs no changes.
    """
    indices = sorted(by_file)
    ref_list = Path(out_dir) / "skani_ref_genomes.txt"
    query_list = Path(out_dir) / "skani_query_list.txt"
    ref_list.write_text("\n".join(db_files[i] for i in indices) + "\n")
    query_list.write_text("\n".join(query_files) + "\n")
    raw = Path(out_dir) / "skani_ani.tsv"

    print(f"Computing query-vs-genome ANI over {len(indices):,} reference genome(s) "
          f"holding {sum(len(v) for v in by_file.values()):,} hit sequence(s)...")
    print("  reference-vs-reference ANI skipped: one genome per species, so no two "
          "references reach 95% ANI")
    # --qi keeps each query sequence a separate entry, since query_hits is keyed by
    # query sequence name. References are deliberately whole files, so no --ri.
    # Screening -s at 90 rather than 95: an initial filter, not the final cutoff.
    run_cmd(["skani", "dist", "-t", threads, "-s", "90",
             "--qi", "--ql", query_list, "--rl", ref_list, "-o", raw])

    by_path = {db_files[i]: i for i in indices}
    query_set = set(query_files)
    query_hits = defaultdict(set)
    unmatched = 0
    with open(raw) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            try:
                ani = float(parts[2])
            except ValueError:
                continue
            if ani < 95.0:
                continue
            # Columns are Ref_file, Query_file, ANI, ..., Ref_name, Query_name.
            # Orientation is read off the query file set rather than assumed, as
            # in the triangle branch. The reference is a whole file here, so its
            # path identifies it - Ref_name would only name its first contig.
            if parts[1] in query_set:
                ref_path, q_name = parts[0], parts[6]
            elif parts[0] in query_set:
                ref_path, q_name = parts[1], parts[5]
            else:
                unmatched += 1
                continue
            file_idx = by_path.get(ref_path)
            if file_idx is None:
                unmatched += 1
                continue
            query_hits[q_name.split()[0]] |= by_file[file_idx]

    if unmatched:
        print(f"  WARNING: {unmatched:,} skani row(s) named a file that is neither a "
              f"query nor a listed reference genome; those self-hits were not applied")
    print(f"  {len(query_hits):,} query sequence(s) matched a reference genome at "
          f">= 95% ANI")
    if not query_hits:
        print("  note: no query sequence is within 95% ANI of any hit genome, so "
              "filter 1 will not drop anything as a self-hit")
    return query_hits


def query_and_ref_triangle(db_files, by_file, query_files, hit_names, threads, out_dir):
    """
    One skani triangle over the query plus the hit reference sequences, for a
    database that is not one genome per species and so needs ref_pairs computed.

    Only the hit sequences are compared, but they are written out one file per
    source genome rather than pooled into a single FASTA. That grouping is load
    bearing: pooling the same sequences into one file shifts skani's ANI by up to
    ~1 here and moves a few percent of pairs across the 95% cutoff, whereas one
    file per genome is byte-identical to comparing the whole genome files and
    roughly 7x faster when a genome contributes one hit out of ten contigs.
    """
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
        # Screening -s at 90 rather than 95: an initial filter, not the final cutoff.
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

ALAMEM_FOOTER = "# ALAMEM END"


def alamem_version():
    """The version string from `alamem --version`, or None if it cannot be read."""
    try:
        out = subprocess.run(["alamem", "--version"], capture_output=True, text=True,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    match = re.search(r"\d+\.\d+\.\d+", out)
    return match.group(0) if match else None


def alamem_header(path):
    """
    alamem's '# key: value' header as a dict, with Rust's Debug quoting stripped.

    Case is preserved deliberately: 'Version' is the binary's version and
    'version' is its --version option, and they differ only in case. Stops at
    the first non-comment line, so the hit rows are never scanned.
    """
    header = {}
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break          # past the header block
                key, sep, value = line[1:].strip().partition(":")
                if sep:
                    header[key.strip()] = value.strip().strip('"')
    except OSError:
        pass
    return header


def alamem_settings_match(header, db_list, query, k, min_len, min_ani):
    """
    True when a result's header records the alignment that is about to be run.

    Numbers are compared as numbers, so 90 and 90.0 agree. Paths are resolved
    against the current directory, since alamem records them as they were given
    on its command line - so resuming from a different working directory
    realigns rather than risking a match on the wrong file. Any key that is
    missing or will not parse counts as a mismatch, on the same principle: a
    header this code does not understand is not one to trust.
    """
    def same_path(recorded, current):
        try:
            return Path(recorded).resolve() == Path(current).resolve()
        except OSError:
            return False

    try:
        return (int(header["kmer_size"]) == int(k)
                and int(header["min_len"]) == int(min_len)
                and float(header["min_ani"]) == float(min_ani)
                and same_path(header["database"], db_list)
                and same_path(header["y_files"], query))
    except (KeyError, ValueError):
        return False


def alamem_completed(path):
    """
    True when the result ends with alamem's footer, i.e. alamem got to the end.

    This is a stronger guarantee than the sidecar config can give: a fingerprint
    says what a file was meant to be, the footer says the file is all there. It
    also survives a copy between machines, which the mtime fingerprint does not.
    Fails closed - an unrecognised or missing footer means realign.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    return any(line.startswith(ALAMEM_FOOTER) for line in tail.splitlines())


def alamem_input_stats(db_list, query):
    """
    Size and mtime of the two inputs - the one thing alamem's header does not
    record. It logs the paths it was given, not their contents, so a query
    regenerated in place under the same name would otherwise be invisible and
    a stale alignment would be reused for the whole run.

    Size and mtime rather than a hash: hashing a database of millions of
    sequences would cost more than the alignment it saves. A '.txt' list of
    FASTA paths is only stat'ed itself, so editing one of the FASTAs it names
    without touching the list is not detected.
    """
    def stat_of(path):
        try:
            st = Path(path).stat()
            return {"path": str(Path(path).resolve()),
                    "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            return {"path": str(path), "size": None, "mtime_ns": None}

    return {"database": stat_of(db_list), "query": stat_of(query)}


def run_alamem(db_list, query, out_path, config_path, threads, min_len, min_ani, k):
    """
    Align the whole query against the streamed database in a single pass.

    The one resumable step in `run`. Alignment is the only part that costs hours
    on a large query, so a finished result is reused when four things hold: the
    footer says alamem finished, its header records the same thresholds and
    paths, its version matches the installed binary, and the inputs have not
    changed on disk since. Anything else realigns.

    Three of those four come from alamem's own header and footer, which is why
    the sidecar config is now only the inputs' size and mtime - the one thing
    the header cannot know, since it records the paths it was given rather than
    their contents.

    The result is still written through atomic(): the footer makes an incomplete
    result detectable, and atomic() keeps one from appearing under the real name
    at all, which matters because the parser below skips malformed rows and a
    truncated result would otherwise look like a smaller valid alignment.
    """
    out_path, config_path = Path(out_path), Path(config_path)
    stats = alamem_input_stats(db_list, query)
    header = alamem_header(out_path) if out_path.exists() else {}

    recorded = None
    if out_path.exists() and config_path.exists():
        try:
            recorded = json.loads(config_path.read_text())
        except ValueError:
            print(f"  WARNING: could not parse {config_path}; realigning")

    # A version bump can change what counts as a hit, so a result from a
    # different alamem is not reusable. Only a *known* mismatch blocks reuse:
    # a result predating the header has no footer either, and the footer check
    # already sends it back through the aligner.
    installed = alamem_version()
    result_version = header.get("Version")
    version_ok = None in (installed, result_version) or installed == result_version
    settings_ok = alamem_settings_match(header, db_list, query, k, min_len, min_ani)

    if (out_path.exists() and alamem_completed(out_path)
            and version_ok and settings_ok and recorded == stats):
        print(f"Reusing {out_path} - alamem {result_version}, same thresholds and inputs")
    else:
        if out_path.exists():
            if not alamem_completed(out_path):
                reason = "has no completion footer, so the run did not finish"
            elif not settings_ok:
                reason = "records different thresholds or input paths"
            elif not version_ok:
                reason = f"was written by alamem {result_version}, not {installed}"
            elif recorded is None:
                reason = "has no matching config recording its inputs"
            else:
                reason = "was built from inputs that have changed on disk"
            print(f"  {out_path} exists but {reason}; realigning")
        print(f"Running alamem against {db_list}...")
        with atomic(out_path) as tmp:
            run_cmd(["alamem", db_list, query, tmp, "-k", k,
                     "-t", threads, "-l", min_len, "--min-ani", min_ani])
            if not alamem_completed(tmp):
                print(f"  WARNING: alamem exited cleanly but wrote no "
                      f"'{ALAMEM_FOOTER}' footer; the result may be incomplete")
        # Written only after alamem succeeds, so the config never vouches for a
        # result that does not exist.
        with atomic(config_path) as tmp:
            tmp.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    hits, malformed = [], 0
    with open(out_path) as f:
        for line in f:
            # alamem's config header and footer are '#' comments, as alasight's
            # own TSVs are. Skipping by prefix rather than by position means the
            # block can grow without this parse having to know how tall it is.
            if not line.strip() or line.startswith("#"):
                continue
            p = [tok.strip() for tok in line.rstrip("\n").split("\t")]
            # 'Reference' is the current column header, 'T name' the pre-0.1.2
            # one; both are uncommented and have 11 fields, so neither is caught
            # by the '#' skip above or by the field count below.
            if p[0] in ("Reference", "T name"):
                continue
            if len(p) != 11:
                malformed += 1    # counted, not ignored: see the warning below
                continue
            try:
                hits.append(AlamemHit(
                    t_name=p[0], q_name=p[1],
                    t_size=int(p[2]), t_start=int(p[3]), t_end=int(p[4]),
                    q_size=int(p[5]), q_start=int(p[6]), q_end=int(p[7]),
                    strand=p[8], ani=float(p[9]), score=int(p[10]),
                ))
            except ValueError:
                # Right field count, wrong types - a renamed column header or a
                # reordered schema. Better counted than raised mid-alignment.
                malformed += 1
    print(f"  {len(hits):,} alamem hit(s)")
    if malformed:
        # Every structural line is accounted for above, so anything left with the
        # wrong field count is a real problem - a truncated file, a stale reused
        # result, or a column added upstream that this parse has not caught up to.
        print(f"  WARNING: {malformed:,} row(s) in {out_path} did not have 11 fields "
              f"and were skipped")
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
    """
    Greedily pair overlapping rows from different clades, largest mutual overlap
    first, each row used at most once.

    The heap holds one entry per row - that row's best remaining partner - rather
    than every candidate pair. The result is the same because every pair is a
    candidate of its lower-indexed row, so the largest pair overall is always
    some row's own best. A stale entry, one whose partner has since been taken,
    can only overstate its overlap, so it surfaces before anything it would
    wrongly outrank and is corrected there. That turns peak memory from one tuple
    per candidate pair - tens of millions on a deep pileup - into one per row.
    """
    n = len(rows)
    if n <= 1:
        return
    rows = sorted(rows, key=lambda r: (int(r[_QS]), int(r[_QE])))
    # Unpacked once. This is the hot loop's only input, and re-parsing these
    # strings inside it costs more than the scanning does.
    starts = [int(r[_QS]) for r in rows]
    ends = [int(r[_QE]) for r in rows]
    species = [r[_RSP] for r in rows]
    leaves = [ref_leaf(index, r[_TNAME]) for r in rows]
    used = [False] * n
    rejected = set()

    def best_partner(i):
        """
        (overlap, j) for row i's largest eligible overlap among later rows, or
        None. Every check here is pure row data - geometry, species string, tree
        leaf - so it needs no cache and no tree lookup; divergence and ANI stay
        at pop time where they are asked once per pair actually taken.
        """
        s1, e1, sp1, leaf1 = starts[i], ends[i], species[i], leaves[i]
        best_ov, best_j = 0, -1
        for j in range(i + 1, n):
            s2 = starts[j]
            if s2 > e1:
                break                     # sorted by start; nothing later overlaps
            if e1 - s2 < best_ov:
                break                     # ceiling on the rest is below what we have
            if used[j]:
                continue
            # Rows are sorted by start, so s2 >= s1 and the overlap's left edge
            # is always s2; only the right edge needs comparing.
            e2 = ends[j]
            ov = (e1 if e1 < e2 else e2) - s2
            if ov <= 0 or ov <= best_ov:
                continue
            sp2 = species[j]
            # 'NA' is not an identity: two references both missing from the tree
            # are not thereby the same clade, so they stay eligible and are
            # arbitrated by ANI at pop time.
            if ((leaf1 == leaves[j] and leaf1 != "NA") or sp1 == sp2) \
                    and "unclassified" not in (sp1, sp2):
                continue                  # same clade, by leaf or by species name
            if rejected and (i, j) in rejected:
                continue
            best_ov, best_j = ov, j
            if best_ov == e1 - s1:
                break                     # full coverage of row i, nothing better
        return (best_ov, best_j) if best_j >= 0 else None

    heap = []
    for i in range(n):
        candidate = best_partner(i)
        if candidate:
            heapq.heappush(heap, (-candidate[0], i, candidate[1]))

    while heap:
        _neg_ov, i, j = heapq.heappop(heap)
        if used[i]:
            continue                      # row i was consumed by a larger overlap
        if used[j] or (rejected and (i, j) in rejected):
            candidate = best_partner(i)   # stale; re-aim row i at what is left
            if candidate:
                heapq.heappush(heap, (-candidate[0], i, candidate[1]))
            continue

        row1, row2 = rows[i], rows[j]
        leaf1, leaf2 = leaves[i], leaves[j]
        div = check_cache(leaf1, leaf2, div_cache)
        if div is None:
            div = tree.divergence(leaf1, leaf2)
            div_cache[(leaf1, leaf2)] = div

        ani = "NA"
        if isinstance(div, str):
            id1, id2 = row1[_TNAME], row2[_TNAME]
            ani = check_cache(id1, id2, ani_cache)
            if ani is None:
                ani = refs_under_95(id1, id2, pairs, index,
                                    starts[i], ends[i], starts[j], ends[j])
                ani_cache[(id1, id2)] = ani

        if not ((not isinstance(div, str) and div >= 1) or ani is True):
            # Too close to be evidence of transfer. Both rows stay available, so
            # record the pair and re-aim row i rather than consuming either.
            rejected.add((i, j))
            candidate = best_partner(i)
            if candidate:
                heapq.heappush(heap, (-candidate[0], i, candidate[1]))
            continue

        out_row = []
        for h in range(len(row1)):
            if h in (0, 1, 9):            # Q name, Q size, Query Species
                out_row.append(row1[h])
            elif h == _QS:
                out_row.append(str(max(starts[i], starts[j])))
            elif h == _QE:
                out_row.append(str(min(ends[i], ends[j])))
            else:
                out_row.append(f"{row1[h]},{row2[h]}")
        out_row += [str(div), str(ani)]

        used[i] = used[j] = True
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


def count_clades(ref_ids, index, pairs):
    """
    Distinct clades among a region's reference sequences.

    References with a tree leaf are counted by leaf, as before. Those without
    one used to collapse to a single clade however unrelated they were, since
    'NA' was doing double duty as an identity. Here they are clustered instead:
    two join the same clade when they come from the same reference genome file,
    or when skani put them at >= 95% ANI. Each remaining cluster counts once.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    leaves, unplaced = set(), []
    for ref_id in ref_ids:
        leaf = ref_leaf(index, ref_id)
        if leaf != "NA":
            leaves.add(leaf)
        else:
            unplaced.append(ref_id)

    # Two contigs of one genome are one organism whatever the ANI says - and with
    # `triangle -i` a pair of short, non-homologous contigs may not be compared
    # at all, which would otherwise split them.
    first_in_file = {}
    for ref_id in unplaced:
        find(ref_id)
        file_idx = ref_file_idx(index, ref_id)
        if file_idx is None:
            continue
        if file_idx in first_in_file:
            union(ref_id, first_in_file[file_idx])
        else:
            first_in_file[file_idx] = ref_id

    for i, a in enumerate(unplaced):
        for b in unplaced[i + 1:]:
            if (min(a, b), max(a, b)) in pairs:
                union(a, b)

    return len(leaves) + len({find(r) for r in unplaced})


def format_summary_row(q_name, iv, index, pairs):
    first = iv["rows"][0]

    ref_ids = [r.strip() for row in iv["rows"]
               for r in str(row.get("T name", "")).split(",") if r.strip()]

    # Divergence times, plus which route each contributing pair came through.
    # A numeric divergence means the tree resolved both reference leaves; the
    # 'unk:' string means at least one was NA and the pair passed on the ANI
    # route instead - which in representatives mode is species distinction with
    # no divergence-time support. Recorded rather than inferred from a blank
    # Avg Divergence Time, since a single tree-supported pair among many ANI
    # ones would otherwise make the whole region look tree-supported.
    div_vals, tree_pairs, ani_pairs = [], 0, 0
    for r in iv["rows"]:
        numeric = False
        for v in str(r.get("Div bt Ref Species", "")).split(","):
            try:
                div_vals.append(float(v.strip()))
                numeric = True
            except ValueError:
                pass
        tree_pairs += numeric
        ani_pairs += not numeric

    if not ani_pairs:
        evidence = "tree"
    elif not tree_pairs:
        evidence = "ani"
    else:
        evidence = "mixed"

    return {
        "Q name": q_name,
        "Q size": first.get("Q size", ""),
        "Q start": iv["start"],
        "Q end": iv["end"],
        "Query Species": first.get("Query Species", ""),
        "Num Regions": len(iv["rows"]),
        "Num Unique Species": count_clades(ref_ids, index, pairs),
        # Empty when every pair in the region came through the ANI route rather
        # than the tree, which is a quick way to spot ANI-only calls.
        "Avg Divergence Time": round(sum(div_vals) / len(div_vals), 4) if div_vals else "",
        "Num Tree Pairs": tree_pairs,
        "Evidence": evidence,
    }


def clustered_regions(overlap_rows, min_size, gap, index, pairs):
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
            summary = format_summary_row(q_name, iv, index, pairs)
            if summary["Num Unique Species"] <= 1:
                continue  # a region needs at least two distinct clades
            regions.append(format_region_row(q_name, iv))
            summaries.append(summary)
    return regions, summaries


# ===========================================================================
# Filter 4 - drop low-complexity regions (DUST)
# ===========================================================================

def read_query_seqs(paths, wanted):
    """
    seq_id -> sequence, for just the ids in `wanted`. Only query sequences that
    survived to a clustered region are held; everything else streams past. Goes
    through open_fasta, so a gzipped query needs no special handling.
    """
    seqs = {}
    for path in paths:
        with open_fasta(path) as handle:
            seq_id, buf = None, []
            for line in handle:
                if line.startswith(">"):
                    if seq_id in wanted:
                        seqs[seq_id] = "".join(buf)
                    seq_id, buf = line[1:].split(None, 1)[0], []
                elif seq_id in wanted:
                    buf.append(line.strip())
            if seq_id in wanted:
                seqs[seq_id] = "".join(buf)
    return seqs


def masked_bases(seq, window, level):
    """
    Bases of `seq` that DUST calls low complexity.

    pydustmasker implements SDUST, the same algorithm as NCBI dustmasker, and
    agrees with `dustmasker -level -window` base for base with one exception: it
    treats N - and any other non-ACGT character - as an ambiguous base rather
    than as low complexity, and never includes one inside a masked interval.
    Adding them back reproduces dustmasker exactly, and an N-rich region is no
    more evidence of transfer than a homopolymer one. Sequences under 4 bp are
    rejected by pydustmasker and dustmasker reports nothing for them, so both
    count as zero.
    """
    from pydustmasker import DustMasker   # only this filter needs it, as with ete3

    if len(seq) < 4:
        return 0
    ambiguous = sum(1 for c in seq if c.upper() not in "ACGT")
    return (DustMasker(seq, window_size=window, score_threshold=level).n_masked_bases
            + ambiguous)


def dust_filter(regions, summaries, query_paths, max_frac, level, window):
    """
    Drop clustered regions that are mostly low complexity, after every other filter.

    Region and summary rows describe the same intervals and are appended in
    lockstep by clustered_regions, so both lists are filtered together and stay
    aligned. Masked fraction is measured over seq[Q start:Q end], the same
    half-open slice size_filter measured, so a region at the size cutoff is
    scored over exactly the bases that got it past that cutoff.

    A region whose query sequence is absent from the FASTA is kept, as is one of
    zero length - both score 0.0, as in the standalone script.
    """
    if max_frac > 1:
        return regions, summaries, 0     # no fraction exceeds 1, so skip the FASTA read

    seqs = read_query_seqs(query_paths, {s["Q name"] for s in summaries})
    kept_regions, kept_summaries, dropped = [], [], 0
    for region, summary in zip(regions, summaries):
        seq = seqs.get(summary["Q name"])
        start, end = int(summary["Q start"]), int(summary["Q end"])
        length = end - start
        frac = 0.0
        if seq is not None and length > 0:
            frac = masked_bases(seq[start:end], window, level) / length
        if frac >= max_frac:
            dropped += 1
            continue
        kept_regions.append(region)
        kept_summaries.append(summary)

    print(f"  {len(kept_summaries):,} of {len(summaries):,} region(s) kept; "
          f"{dropped:,} dropped at >= {max_frac:g} masked")
    return kept_regions, kept_summaries, dropped


# ===========================================================================
# Pipeline
# ===========================================================================

def write_region_pair(out_dir, name, stem, regions, summaries, header):
    """
    Write a region table and its summary. Called once per step that produces
    regions, so each filter's output stays on disk like every other step's.
    """
    region_cols = FIXED_COLS + META_COLS
    write_tsv(out_dir / f"{name}_{stem}.tsv", region_cols,
              [[r[c] for c in region_cols] for r in regions], header)
    write_tsv(out_dir / f"{name}_{stem}_summary.tsv", SUMMARY_COLS,
              [[s[c] for c in SUMMARY_COLS] for s in summaries], header)

def cmd_run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(args.database)
    name = out_dir.name
    header = config_header_text(args)

    require_skani()
    import pydustmasker  # noqa: F401 - fail fast before alamem, not after

    # -tr is optional: the database records the tree it was built against, and
    # the index's leaf names only mean anything relative to that tree.
    config = read_db_config(db_dir)
    tree_dir = args.tree or config.get("tree")
    if not tree_dir:
        sys.exit(f"No tree directory. Pass -tr, or rebuild {db_dir} so it records one.")
    if args.tree and config.get("tree") and Path(args.tree).resolve() != Path(config["tree"]):
        print(f"  WARNING: -tr {args.tree} is not the tree this database was built against "
              f"({config['tree']}). The index's leaf names came from that tree.")

    # Recorded at build time, since it is a property of the reference set rather
    # than of a query; the flag only overrides that record. Not inferred, because
    # skipping the reference-vs-reference ANI is only sound for a database that
    # really does hold one genome per species.
    representatives = (args.ref_representatives if args.ref_representatives is not None
                       else bool(config.get("representatives")))

    print("Start time:", datetime.now())
    print(f"Query: {args.query}\nDatabase: {db_dir}\nTree: {tree_dir}\n"
          f"Threads: {args.threads}\n"
          f"One genome per species: {representatives}\n")

    tree = DivergenceTree(tree_dir)
    index = load_index(db_dir / DB_INDEX)

    query_paths = resolve_fasta_inputs(args.query)
    query_ids = [seq_id for seq_id, _ in iter_fasta_headers(query_paths)]
    print(f"Read {len(query_ids)} query sequence(s) from {len(query_paths)} file(s)")
    query_species = resolve_query_species(args, query_ids)

    hits = run_alamem(db_dir / DB_FASTA_LIST, args.query,
                      out_dir / f"{name}_alamem_results.tsv",
                      out_dir / f"{name}_alamem_config.json",
                      args.threads, args.min_len, args.min_ani, args.kmer_size)

    # After alamem, so only the references it actually hit are compared.
    query_hits, ref_pairs = run_skani_ani(
        db_dir, [str(p) for p in query_paths],
        {h.t_name for h in hits}, index, args.threads, out_dir, representatives)

    print("Filtering out references too similar to the query...")
    first_div = first_divergence_filter(hits, query_species, index,
                                        args.min_ani, query_hits)
    write_tsv(out_dir / f"{name}_first_div_output.tsv", FIRST_DIV_HEADER, first_div, header)

    print("Pairing overlapping hits...")
    overlap = find_overlap_and_div_max(compress(first_div), tree, ref_pairs, index)
    write_tsv(out_dir / f"{name}_overlap_div.tsv", OVERLAP_HEADER, overlap, header)

    print("Building clustered regions...")
    regions, summaries = clustered_regions(overlap, args.size_filter, args.cluster_size,
                                           index, ref_pairs)
    write_region_pair(out_dir, name, "clustered_regions", regions, summaries, header)

    # Filter 4 - low complexity, last so it only sees regions everything else kept.
    # Its own pair of files, so the pre-DUST regions above survive on disk.
    print("Filtering low-complexity regions...")
    regions, summaries, n_dust = dust_filter(regions, summaries, query_paths,
                                             args.max_masked_frac,
                                             args.dustmasker_level,
                                             args.dustmasker_window)
    dust_header = header + (f"# dust filter: dropped {n_dust} region(s) with masked "
                            f"fraction >= {args.max_masked_frac}\n#\n")
    write_region_pair(out_dir, name, "dust_regions", regions, summaries, dust_header)

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
        "representatives": bool(getattr(args, "representatives", False)),
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


def build_index(db_dir, species_file, tree_dir, representatives=False):
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
    seqs_by_route = defaultdict(int)
    species_by_route = defaultdict(int)
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
                leaf_cache[species] = tree.leaf_match(species)
                species_by_route[leaf_cache[species][1]] += 1
            leaf, route = leaf_cache[species]
            seqs_by_route[route] += 1
            out.write(f"{seq_id}\t{species}\t{length}\t{leaf}\t{file_idx}\n")
            total += 1
            if seq_id in species_map:
                covered += 1
            if leaf == "NA":
                unusable += 1

    print(f"  index written to {out_path}")
    print(f"  {total:,} sequences | {covered:,} found in the species file "
          f"({100.0 * covered / total if total else 0:.1f}%) | "
          f"{total - unusable:,} with a usable tree leaf "
          f"({100.0 * (total - unusable) / total if total else 0:.1f}%)")
    print(f"  {len(leaf_cache):,} distinct species, "
          f"{sum(1 for leaf, _ in leaf_cache.values() if leaf == 'NA'):,} "
          f"of which are not in the tree")

    # Per-route counts, because the sequence-weighted percentage above is
    # dominated by whichever species happen to have the most contigs.
    print("  match route      species     sequences")
    for route in ("species", "genus_species", "genus", "genus_member",
                  "no_match", "unclassified", "unnamed"):
        if species_by_route[route]:
            print(f"  {route:<14} {species_by_route[route]:>9,} {seqs_by_route[route]:>13,}")

    unlabelled = seqs_by_route["unclassified"] + seqs_by_route["unnamed"]
    labelled = total - unlabelled
    # First, because an ID mismatch also trips the two warnings below and this is
    # the one that names the cause.
    if total and covered / total < 0.5:
        print("  WARNING: over half the references are missing from the species file. "
              "Check that its first column holds sequence IDs, not assembly accessions.")
    if labelled and (total - unusable) / labelled < 0.5:
        if representatives:
            # Not a warning here: with one genome per species, ref_pairs is empty,
            # so a pair with no tree leaf passes on species distinction instead of
            # being dropped. It just arrives with no divergence time attached.
            print("  NOTE: over half the labelled references have no tree leaf. With one "
                  "genome per species those pairs still pass, on species distinction "
                  "alone, but carry no divergence time - the Evidence column in the "
                  "region summary says which route each region used.")
        else:
            print("  WARNING: over half the labelled references have no tree leaf and cannot "
                  "contribute to the pairing step, so most regions will be dropped.")
    if total and unlabelled / total > 0.25:
        print(f"  WARNING: {100.0 * unlabelled / total:.0f}% of references have no species "
              f"label at all. They can still pair through the ANI route, with no "
              f"divergence-time support.")


def cmd_build_db(args):
    db_dir = Path(args.database)
    db_dir.mkdir(parents=True, exist_ok=True)

    #require_skani()
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
    build_index(db_dir, args.species_file, args.tree, args.representatives)

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
    run.add_argument("--ref-representatives", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="the database holds one genome per species, so no two references "
                          "reach 95%% ANI and the reference-vs-reference ANI can be skipped "
                          "(default: whatever build-db recorded)")
    run.add_argument("-k", "--kmer-size", type=int, default=11,
                     help="alamem seed k-mer size (default: 11, matching alamem's own "
                          "default); larger is faster and less sensitive")
    run.add_argument("--min-len", type=int, default=40,
                     help="minimum alamem hit length in bp (default: 40)")
    run.add_argument("--min-ani", type=float, default=90.0,
                     help="minimum percent identity / ANI of a hit (default: 90)")
    run.add_argument("--size-filter", type=int, default=150,
                     help="drop clustered regions smaller than this many bp (default: 150)")
    run.add_argument("--cluster-size", type=int, default=0,
                     help="merge clustered regions within this many bp (default: 0)")
    run.add_argument("--max-masked-frac", type=float, default=0.5,
                     help="drop a clustered region when this fraction or more of its bases are "
                          "DUST-masked; any value above 1 disables the filter (default: 0.5)")
    run.add_argument("--dustmasker-level", type=int, default=20,
                     help="DUST score threshold, as dustmasker -level (default: 20)")
    run.add_argument("--dustmasker-window", type=int, default=64,
                     help="DUST window size, as dustmasker -window (default: 64)")
    run.set_defaults(func=cmd_run)

    build = sub.add_parser("build-db", help="build an ALASIGHT database")
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
    build.add_argument("--representatives", action="store_true",
                       help="the reference set holds one genome per species, e.g. the GTDB "
                            "species representatives; recorded in the database so `run` can "
                            "skip the reference-vs-reference ANI entirely")
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
