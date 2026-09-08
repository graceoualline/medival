#!/usr/bin/env python3
"""
ALASIGHT - find horizontally transferred regions in a query genome.

A query FASTA is aligned against a reference database with alamem. Hits to the
query's own close relatives are dropped, then overlapping hits are paired and
kept only where the two reference species are separated by enough divergence
time on the Time Tree of Life. Regions are read off query positions covered by
two or more such clades, rather than by pairing hits against each other.

Three subcommands:

  build-tree preprocess a TimeTree newick into the Euler tour and range-minimum
             tables the divergence lookups need. Optional: build-db does this
             itself when -tr is not already prepared. Useful on its own when the
             machine that has ete3 is not the one building databases.
  build-db   one-off construction of an ALASIGHT database. Resumable: every step
             is skipped if its output already exists. It runs no aligner, and
             runs skani only for --skani-sketch, which prebuilds the reference
             sketches so `run` searches them instead of re-sketching per query.
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
  <name>_clustered_regions_depth.tsv   those regions cut at every depth change
  <name>_dust_regions.tsv              regions left after DUST masking
  <name>_dust_regions_summary.tsv      one row per surviving region
  <name>_dust_regions_depth.tsv        those regions cut at every depth change

The two _depth.tsv files are what --no-depth turns off. A summary row says only
that some base of the region had two or more clades on it; a depth row says how
many, over a stretch where that number does not change, so the regions can be
drawn as a coverage track rather than as a flat shaded block.

Coordinates are half-open throughout: a hit's Q end, and a region's, is one
past the last base, so an interval covers start .. end-1 and its length is
end - start. Two intervals that abut share no base.

Examples:
  alasight.py build-db -i genomes.txt -d gtdb_db -tr timetree/ \\
             -n "TimeTree v5 Final.nwk" --species-file sp.tsv
  alasight.py build-tree -n "TimeTree v5 Final.nwk" -o timetree/   # separately, if preferred
  alasight.py run -q genome.fa.gz -o results -d gtdb_db -t 32

Wherever a FASTA is expected you may instead pass a .txt file containing one
FASTA path per line, and any of those files may be gzipped.
"""

import argparse
import bisect
import contextlib
import csv
import gzip
import json
import math
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
DB_SKANI_SKETCH = "skani_sketch"

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
# Reference-side columns of a region, one token per contributing hit.
REGION_META_COLS = [
    "T name", "T size", "T start", "T end", "Percent Identity",
    "Reference Species", "Clade",
]

# Positions within a FIRST_DIV_HEADER row, used by the overlap pairing.
_QS = 2       # Q start
_QE = 3       # Q end
_TNAME = 4    # T name
_RSP = 10     # Reference Species

FIXED_COLS = ["Q name", "Q size", "Q start", "Q end", "Query Species"]
# How much support a region has, and of what kind.
SUMMARY_COLS = FIXED_COLS + ["Num Hits", "Num Clades", "Peak Clades",
                             "Avg Divergence Time", "Max Divergence Time", "Evidence"]
# A region cut into stretches of constant depth. "Q start"/"Q end" are the
# stretch; "Region Q start"/"Region Q end" are the region it came out of.
DEPTH_COLS = FIXED_COLS + ["Region Q start", "Region Q end", "Region Index",
                           "Depth", "Log2 Depth", "Num Clades", "Num Hits", "Clades"]

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
    """
    Size of the overlap between two half-open intervals, or None.

    Half-open because a hit's Q end and T end are one past the last base, so
    intervals that merely abut share nothing. Returning None rather than 0 for
    that case makes no difference to the one caller - crude_ani_overlap turns
    both into an ANI below 95 - but it keeps the answer honest.
    """
    if max(start1, start2) < min(end1, end2):
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
                  representatives=False, sketch_db=None):
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
        return query_vs_genomes(db_files, by_file, query_files, threads, out_dir,
                                sketch_db), set()
    return query_and_ref_triangle(db_files, by_file, query_files, hit_names,
                                  threads, out_dir)


def query_vs_genomes(db_files, by_file, query_files, threads, out_dir, sketch_db=None):
    """
    query_hits[q_name] -> every hit sequence belonging to a reference genome that
    is the same organism as that query sequence, i.e. >= 95% ANI to it.

    Whole reference genomes are compared, not individual hit contigs, because
    that is the granularity the question is asked at: with one genome per species
    the organism *is* the genome, and 95% ANI is where GTDB draws the species
    boundary to begin with. No hit sequences need extracting, so the per-genome
    scratch files and the -i grouping question both go away, and a short contig
    skani cannot sketch can no longer go missing from the output and be read as
    "not a self-hit" - a whole genome always sketches.

    With `sketch_db`, `skani search` reads sketches built once at build time
    instead of re-sketching the hit genomes on every query. That searches the
    whole database rather than only the genomes alamem hit, which is why matches
    to genomes with no hits are expected below rather than anomalous.

    A self-match is expanded back to every hit sequence in that genome before
    returning, so filter 1 keeps comparing sequence names and needs no changes.
    """
    query_list = Path(out_dir) / "skani_query_list.txt"
    query_list.write_text("\n".join(query_files) + "\n")
    raw = Path(out_dir) / "skani_ani.tsv"
    n_hit_seqs = sum(len(v) for v in by_file.values())

    if sketch_db:
        print(f"Searching {len(db_files):,} prebuilt reference sketch(es) for the query; "
              f"{n_hit_seqs:,} hit sequence(s) across {len(by_file):,} genome(s)...")
        print("  reference-vs-reference ANI skipped: one genome per species, so no two "
              "references reach 95% ANI")
        # --qi keeps each query sequence a separate entry, since query_hits is
        # keyed by query sequence name.
        run_cmd(["skani", "search", "-t", threads, "-d", sketch_db,
                 "--qi", "--ql", query_list, "-o", raw])
    else:
        indices = sorted(by_file)
        ref_list = Path(out_dir) / "skani_ref_genomes.txt"
        ref_list.write_text("\n".join(db_files[i] for i in indices) + "\n")
        print(f"Computing query-vs-genome ANI over {len(indices):,} reference genome(s) "
              f"holding {n_hit_seqs:,} hit sequence(s)...")
        print("  reference-vs-reference ANI skipped: one genome per species, so no two "
              "references reach 95% ANI")
        # Screening -s at 90 rather than 95: an initial filter, not the final cutoff.
        run_cmd(["skani", "dist", "-t", threads, "-s", "90",
                 "--qi", "--ql", query_list, "--rl", ref_list, "-o", raw])

    # Every database genome, not only the hit ones: a search covers the whole
    # database, and a match to a genome that contributed no hits is a no-op
    # rather than something to warn about.
    by_path = {path: i for i, path in enumerate(db_files)}
    query_set = set(query_files)
    query_hits = defaultdict(set)
    no_hits = unknown = 0
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
            # Orientation is read off the query file set rather than assumed.
            # The reference is a whole file here, so its path identifies it -
            # Ref_name would only name its first contig.
            if parts[1] in query_set:
                ref_path, q_name = parts[0], parts[6]
            elif parts[0] in query_set:
                ref_path, q_name = parts[1], parts[5]
            else:
                unknown += 1
                continue
            file_idx = by_path.get(ref_path)
            if file_idx is None:
                unknown += 1
                continue
            hits = by_file.get(file_idx)
            if not hits:
                no_hits += 1          # same organism, but it contributed no hits
                continue
            query_hits[q_name.split()[0]] |= hits

    if unknown:
        print(f"  WARNING: {unknown:,} skani row(s) named a file that is neither a query "
              f"nor a database reference; those self-hits were not applied")
    print(f"  {len(query_hits):,} query sequence(s) matched a reference genome at "
          f">= 95% ANI" + (f"; {no_hits:,} match(es) were to genomes with no alamem hits"
                           if no_hits else ""))
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
# Filter 2 - supported regions, by sweep over query positions
# ===========================================================================

def clade_of(index, t_name):
    """
    The clade a reference hit belongs to.

    Its tree leaf when it has one, since divergence is measured between leaves;
    otherwise its source genome file, because one file is one organism - and in
    a one-genome-per-species database, one species. Tagged rather than returned
    bare so a leaf name and a file index can never collide.
    """
    leaf = ref_leaf(index, t_name)
    return ("leaf", leaf) if leaf != "NA" else ("file", ref_file_idx(index, t_name))


def merge_close_clades(clades, tree, pairs, index):
    """
    clade -> group, merging clades too close together to be evidence of transfer.

    Two clades are distinct when the tree separates them by at least 1 MYA, or -
    where either has no leaf, so no divergence to read - when their reference
    sequences are below 95% ANI. That is the same test the old pairwise filter
    applied, lifted from pairs of overlapping hits to pairs of clades. It now
    runs once per pair of distinct clades instead of once per pair of
    overlapping hits, which is what makes a deep pileup affordable: the cost is
    set by how many clades were hit, not by how many times they were hit.
    """
    parent = {c: c for c in clades}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    leaves = [c for c in clades if c[0] == "leaf"]
    for i, a in enumerate(leaves):
        for b in leaves[i + 1:]:
            div = tree.divergence(a[1], b[1])
            if not isinstance(div, str) and div < 1:
                union(a, b)

    # Clades with no leaf are an ANI question instead, and ref_pairs already
    # holds every reference sequence pair at >= 95%.
    for id1, id2 in pairs:
        c1, c2 = clade_of(index, id1), clade_of(index, id2)
        if c1 in parent and c2 in parent:
            union(c1, c2)

    return {c: find(c) for c in clades}


def sweep_spans(hits, min_size, gap):
    """
    (start, end, peak) for every query interval covered by at least two clade
    groups, merged across gaps of up to `gap` and dropping anything shorter
    than `min_size`.

    `hits` is (start, end, group) per contributing hit, half-open [start, end)
    because a hit's Q end is one past its last base. Only a counter of active
    groups is maintained, so each hit costs two events and one dict update
    however deep the pileup gets. `peak` is the most groups covering any single
    base in the span - the difference between two clades brushing past each
    other and eight of them stacked on the same locus.

    Every event at a coordinate is applied before the counter is read, so
    `live` is the number of groups covering that base and nothing else. Two
    consequences, both wrong in the older event-at-a-time version: hits that
    merely abut do not count as overlapping, so no span opens between them and
    no peak is inflated by one; and a clade leaving exactly where another
    arrives does not split a span, because the counter never dips.
    """
    events = defaultdict(list)
    for start, end, group in hits:
        if end <= start:
            continue                         # covers no base, so no events
        events[start].append((group, 1))
        events[end].append((group, -1))

    active = defaultdict(int)
    live = peak = 0
    spans, span_start = [], None
    for pos in sorted(events):
        for group, delta in events[pos]:
            if delta > 0:
                active[group] += 1
                if active[group] == 1:
                    live += 1
            else:
                active[group] -= 1
                if active[group] == 0:
                    live -= 1
                    del active[group]        # so len(active) == live
        if live >= 2:
            if span_start is None:
                span_start, peak = pos, live
            else:
                peak = max(peak, live)
        elif span_start is not None:
            spans.append((span_start, pos, peak))
            span_start = None

    merged = []
    for start, end, span_peak in spans:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], span_peak)
        else:
            merged.append([start, end, span_peak])
    return [(s, e, pk) for s, e, pk in merged if e - s >= min_size]


def _divergence_stats(groups, tree):
    """
    (avg, max, evidence) over the distinct pairs of clade groups in a region.

    A group is named by a leaf only if one of its clades had one; pairs where
    either side does not are the ones that passed on ANI rather than on
    divergence, and 'evidence' records that mix so an ANI-only call is not
    mistaken for a tree-supported one.
    """
    leaves = sorted({g[1] for g in groups if g[0] == "leaf"})
    values = []
    tree_pairs = ani_pairs = 0
    for i, a in enumerate(leaves):
        for b in leaves[i + 1:]:
            div = tree.divergence(a, b)
            if isinstance(div, str):
                ani_pairs += 1
            else:
                values.append(div)
                tree_pairs += 1
    n_groups = len(groups)
    ani_pairs += n_groups * (n_groups - 1) // 2 - (tree_pairs + ani_pairs)

    if not ani_pairs:
        evidence = "tree"
    elif not tree_pairs:
        evidence = "ani"
    else:
        evidence = "mixed"
    if not values:
        return "", "", evidence
    return round(sum(values) / len(values), 4), round(max(values), 4), evidence


def supported_regions(rows, tree, pairs, index, min_size, gap):
    """
    Regions of the query covered by hits from two or more distinct clades.

    Positions, not pairs of hits. The two are equivalent - a base covered by two
    hits from distinct clades lies in that pair's overlap, and every base of
    such an overlap is covered by both - but the old pairing consumed each hit
    into at most one pair, so a third hit overlapping only one member of a
    chosen pair could support no region at all and its interval was lost.

    Returns (region_rows, summary_rows, group) - the first two parallel lists of
    dicts, one region per row, the hits that support it, and how much support
    that is. `group` is the clade -> group map, handed back so depth_rows can
    reuse it instead of paying for merge_close_clades a second time.
    """
    by_query = defaultdict(list)
    for row in rows:
        by_query[row[0]].append(row)

    clades = {clade_of(index, row[_TNAME]) for row in rows}
    group = merge_close_clades(clades, tree, pairs, index)
    print(f"  {len(clades):,} clade(s) among the hits, {len(set(group.values())):,} "
          f"after merging any too close to separate")

    regions, summaries = [], []
    for q_name, qrows in by_query.items():
        groups = [group[clade_of(index, r[_TNAME])] for r in qrows]
        starts = [int(r[_QS]) for r in qrows]
        ends = [int(r[_QE]) for r in qrows]
        spans = sweep_spans(list(zip(starts, ends, groups)), min_size, gap)
        if not spans:
            continue

        # Second pass for the contributing hits. Kept out of the sweep on
        # purpose: accumulating them per event costs O(pileup depth) each time
        # and puts the quadratic straight back in.
        #
        # Half-open on both sides: hit [s, e) meets span [S, E) only where
        # S < e and s < E, so a hit that stops exactly at a span's start, or
        # starts exactly at its end, shares no base with it and is not a member.
        span_starts = [s for s, _e, _pk in spans]
        members = [[] for _ in spans]
        for i in range(len(qrows)):
            j = bisect.bisect_left(span_starts, ends[i]) - 1
            while j >= 0 and spans[j][1] > starts[i]:
                if spans[j][0] < ends[i]:
                    members[j].append(i)
                j -= 1

        for (start, end, peak), member in zip(spans, members):
            if not member:
                continue
            first = qrows[member[0]]
            region = {
                "Q name": q_name,
                "Q size": first[1],
                "Q start": start,
                "Q end": end,
                "Query Species": first[9],
            }
            for col, position in (("T name", _TNAME), ("T size", 5),
                                  ("T start", 6), ("T end", 7),
                                  ("Percent Identity", 8), ("Reference Species", _RSP)):
                region[col] = "|".join(qrows[i][position] for i in member)
            present = {groups[i] for i in member}
            region["Clade"] = "|".join(
                f"{groups[i][0]}:{groups[i][1]}" for i in member)
            avg, mx, evidence = _divergence_stats(present, tree)
            regions.append(region)
            summaries.append({
                "Q name": q_name,
                "Q size": first[1],
                "Q start": start,
                "Q end": end,
                "Query Species": first[9],
                "Num Hits": len(member),
                "Num Clades": len(present),
                "Peak Clades": peak,
                "Avg Divergence Time": avg,
                "Max Divergence Time": mx,
                "Evidence": evidence,
            })

    print(f"  {len(summaries):,} supported region(s)")
    return regions, summaries, group


# ===========================================================================
# Depth of coverage
# ===========================================================================

def depth_track(intervals):
    """
    A whole query's coverage track: [(start, end, groups, hits), ...].

    `intervals` is (start, end, group, hit_id) per hit, half-open [start, end)
    like everything else, so a segment's length is end - start - the same
    measure size_filter and the DUST filter apply to a region. Segments are the
    stretches between consecutive hit boundaries, and nothing changes inside one
    by construction, so `groups` and `hits` are the exact sets covering every
    base of it rather than a maximum or an average. Uncovered stretches are left
    out; the caller fills them in against the region it is cutting.

    This keeps the covering sets where sweep_spans deliberately keeps only a
    counter, so it costs O(pileup depth) per segment instead of O(1). That is
    the price of a per-base answer, and it buys a track whose size is the size
    of the file it is written to.
    """
    events = defaultdict(list)
    for start, end, grp, hit_id in intervals:
        if end <= start:
            continue                       # covers no base, so no events
        events[start].append((grp, hit_id, 1))
        events[end].append((grp, hit_id, -1))

    active_groups = defaultdict(int)
    active_hits = set()
    track, prev = [], None
    for pos in sorted(events):
        if prev is not None and active_hits:
            track.append((prev, pos,
                          frozenset(active_groups), frozenset(active_hits)))
        for grp, hit_id, delta in events[pos]:
            if delta == 1:
                active_groups[grp] += 1
                active_hits.add(hit_id)
            else:
                active_groups[grp] -= 1
                if active_groups[grp] == 0:
                    del active_groups[grp]  # so the keys are the live groups
                active_hits.discard(hit_id)
        prev = pos
    return track


def depth_rows(rows, group, index, summaries):
    """
    Every region in `summaries` cut into maximal runs of constant depth.

    Depth is how many distinct clade groups cover a base, the per-base quantity
    Peak Clades reports the maximum of. A region is one summary row however its
    depth varies inside it, so a picture drawn from the summary can only shade
    "two or more clades somewhere in here". These rows carry the number itself,
    one row per stretch over which it does not change.

    Cuts fall where the depth changes and at the region's own edges, so a
    region's rows tile it exactly - no gaps, no overlap, lengths summing to
    Q end - Q start. Depth 1, or 0, can appear inside a region that
    --cluster-size merged across a gap: those rows are kept, because a bridged
    gap is exactly what a depth chart is worth drawing for. Log2 Depth is blank
    at depth 0.

    Depth counts groups covering every base of a run; Num Clades counts groups
    appearing anywhere in it, so the two differ where clades swap in and out
    across a cut that did not change the total. Num Hits counts hits overlapping
    the run, as it does per region in the summary.

    A region of zero length gets no rows, having no bases to describe, though
    sweep_spans no longer emits one.

    The deepest row of a region equals the Peak Clades on its summary row, by
    construction: sweep_spans counts the groups covering a base under the same
    half-open reading, so the two cannot disagree.
    """
    by_query = defaultdict(list)
    for row in rows:
        by_query[row[0]].append(row)

    # Grouped by query and ascending within it, which is the order
    # supported_regions appended them in, so the output follows the summary.
    by_region = defaultdict(list)
    for i, summary in enumerate(summaries):
        by_region[summary["Q name"]].append(i)

    out = []
    for q_name, region_idxs in by_region.items():
        qrows = by_query.get(q_name, [])
        track = depth_track(
            (int(r[_QS]), int(r[_QE]), group[clade_of(index, r[_TNAME])], i)
            for i, r in enumerate(qrows))
        ends = [seg[1] for seg in track]

        for region_i in region_idxs:
            summary = summaries[region_i]
            r_start, r_end = int(summary["Q start"]), int(summary["Q end"])

            # Walk the region, taking each segment that overlaps it and calling
            # anything between them depth 0.
            pieces = []
            pos = r_start
            j = bisect.bisect_right(ends, r_start)
            while pos < r_end:
                if j >= len(track) or track[j][0] >= r_end:
                    pieces.append([pos, r_end, 0, set(), set()])
                    break
                seg_start, seg_end, groups, hits = track[j]
                if seg_start > pos:
                    pieces.append([pos, seg_start, 0, set(), set()])
                    pos = seg_start
                stop = min(seg_end, r_end)
                pieces.append([pos, stop, len(groups), set(groups), set(hits)])
                pos = stop
                j += 1

            merged = []
            for start, stop, depth, groups, hits in pieces:
                if merged and merged[-1][2] == depth:
                    merged[-1][1] = stop
                    merged[-1][3] |= groups
                    merged[-1][4] |= hits
                else:
                    merged.append([start, stop, depth, groups, hits])

            for start, stop, depth, groups, hits in merged:
                out.append({
                    "Q name":         q_name,
                    "Q size":         summary["Q size"],
                    "Q start":        start,
                    "Q end":          stop,
                    "Query Species":  summary["Query Species"],
                    "Region Q start": r_start,
                    "Region Q end":   r_end,
                    "Region Index":   region_i,
                    "Depth":          depth,
                    "Log2 Depth":     round(math.log2(depth), 4) if depth else "",
                    "Num Clades":     len(groups),
                    "Num Hits":       len(hits),
                    "Clades":         "|".join(sorted(f"{k}:{v}" for k, v in groups)),
                })

    print(f"  {len(out):,} constant-depth row(s) over {len(summaries):,} region(s)")
    return out


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
    region_cols = FIXED_COLS + REGION_META_COLS
    write_tsv(out_dir / f"{name}_{stem}.tsv", region_cols,
              [[r[c] for c in region_cols] for r in regions], header)
    write_tsv(out_dir / f"{name}_{stem}_summary.tsv", SUMMARY_COLS,
              [[s[c] for c in SUMMARY_COLS] for s in summaries], header)


def write_depth(out_dir, name, stem, rows, group, index, summaries, header):
    """Write the depth track for one step's regions, beside its summary."""
    depth = depth_rows(rows, group, index, summaries)
    write_tsv(out_dir / f"{name}_{stem}_depth.tsv", DEPTH_COLS,
              [[d[c] for c in DEPTH_COLS] for d in depth], header)


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

    # Recorded at build time like the tree, but checked on disk too, so a
    # database sketched after it was built is still picked up.
    sketch_db = config.get("skani_sketch") or None
    if sketch_db and not Path(sketch_db).exists():
        print(f"  WARNING: recorded skani sketch database {sketch_db} is missing; "
              f"re-sketching the hit genomes for this run instead")
        sketch_db = None
    if not sketch_db and (db_dir / DB_SKANI_SKETCH).exists():
        sketch_db = str(db_dir / DB_SKANI_SKETCH)
    built_by = config.get("skani_sketch_version")
    installed = skani_version()
    if sketch_db and built_by and installed and ".".join(map(str, installed)) != built_by:
        print(f"  WARNING: the sketch database was built by skani {built_by} but "
              f"{'.'.join(map(str, installed))} is installed; sketch parameters and "
              f"format are fixed at sketch time, so rebuild it if results look wrong.")

    print("Start time:", datetime.now())
    print(f"Query: {args.query}\nDatabase: {db_dir}\nTree: {tree_dir}\n"
          f"Threads: {args.threads}\n"
          f"One genome per species: {representatives}\n"
          f"Prebuilt skani sketches: {sketch_db or 'none'}\n")

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
        {h.t_name for h in hits}, index, args.threads, out_dir, representatives,
        sketch_db)

    print("Filtering out references too similar to the query...")
    first_div = first_divergence_filter(hits, query_species, index,
                                        args.min_ani, query_hits)
    write_tsv(out_dir / f"{name}_first_div_output.tsv", FIRST_DIV_HEADER, first_div, header)

    print("Finding supported regions...")
    regions, summaries, group = supported_regions(first_div, tree, ref_pairs, index,
                                                  args.size_filter, args.cluster_size)
    write_region_pair(out_dir, name, "clustered_regions", regions, summaries, header)
    if args.depth:
        print("Building depth track...")
        write_depth(out_dir, name, "clustered_regions", first_div, group, index,
                    summaries, header)

    # Filter 3 - low complexity, last so it only sees regions everything else kept.
    # Its own pair of files, so the pre-DUST regions above survive on disk.
    print("Filtering low-complexity regions...")
    regions, summaries, n_dust = dust_filter(regions, summaries, query_paths,
                                             args.max_masked_frac,
                                             args.dustmasker_level,
                                             args.dustmasker_window)
    dust_header = header + (f"# dust filter: dropped {n_dust} region(s) with masked "
                            f"fraction >= {args.max_masked_frac}\n#\n")
    write_region_pair(out_dir, name, "dust_regions", regions, summaries, dust_header)
    if args.depth:
        write_depth(out_dir, name, "dust_regions", first_div, group, index,
                    summaries, dust_header)

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


def build_skani_sketch(db_dir, fasta_list, threads):
    """
    Sketch every reference once, into a database `run` can search against.

    Sketching is what dominates the per-query ANI step: `dist` reads and sketches
    every hit genome on every run, which for a GTDB-sized reference set is
    hundreds of Gbp of identical work each time. Doing it once here turns that
    into loading precomputed sketches, and `skani search` screens on marker
    sketches so only real candidates are loaded in full.

    Searching the whole database rather than just the hit genomes is the trade:
    more candidates, all marker-screened, against no sketching at all. It also
    makes the comparison independent of which references alamem happened to hit.
    """
    out_dir = Path(db_dir) / DB_SKANI_SKETCH
    if out_dir.exists():
        print(f"Skipping skani sketch database, {out_dir} exists")
        return out_dir
    require_skani()
    print(f"Sketching {fasta_list} into {out_dir}...")
    # Through atomic() so an interrupted sketch leaves no directory for the next
    # build to mistake for a finished one.
    with atomic(out_dir) as tmp:
        run_cmd(["skani", "sketch", "-t", threads, "-l", fasta_list, "-o", tmp])
    print(f"  sketch database written to {out_dir}")
    return out_dir


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
        # The sketch database's parameters are baked in at sketch time, so the
        # version that built it is recorded: a later skani whose defaults or
        # sketch format differ would otherwise be used against it silently.
        "skani_sketch": str((Path(db_dir) / DB_SKANI_SKETCH).resolve())
                        if (Path(db_dir) / DB_SKANI_SKETCH).exists() else None,
        "skani_sketch_version": skani_version() and ".".join(
            map(str, skani_version())) if (Path(db_dir) / DB_SKANI_SKETCH).exists() else None,
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
    if args.skani_sketch:
        build_skani_sketch(db_dir, db_dir / DB_FASTA_LIST, args.threads)

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
    run.add_argument("-k", "--kmer-size", type=int, default=15,
                     help="alamem seed k-mer size (default: 15, larger than alamem for large database); larger is less sensitive but reduces false positives for large databases")
    run.add_argument("--min-len", type=int, default=40,
                     help="minimum alamem hit length in bp (default: 40)")
    run.add_argument("--min-ani", type=float, default=90.0,
                     help="minimum percent identity / ANI of a hit (default: 90)")
    run.add_argument("--size-filter", type=int, default=150,
                     help="drop clustered regions smaller than this many bp (default: 150)")
    run.add_argument("--cluster-size", type=int, default=0,
                     help="merge clustered regions within this many bp (default: 0)")
    run.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True,
                     help="also write a clade-depth track beside each region summary, one "
                          "row per stretch of constant depth within a region (default: on)")
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
    build.add_argument("--skani-sketch", action="store_true",
                       help="prebuild a skani sketch database in the database directory, so "
                            "`run` searches precomputed sketches instead of re-sketching "
                            "every hit genome on each query")
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
