#!/usr/bin/env python3
"""
Build an ALASIGHT --species-file directly from GTDB metadata.

GTDB assigns every genome a species, so for GTDB references the label is a
lookup, not a prediction. Every sequence in a genome file inherits that genome's
species; no per-contig classification is involved. This matters beyond accuracy:
GTDB species are ~95% ANI clusters, which is the same question refs_under_95
asks, so canonical labels make much of that ANI arbitration redundant.

Genomes are matched to metadata by assembly accession, taken from the FASTA
filename, in three tiers - exact, GCA/GCF swapped, then digits only. Each match
records the tier that produced it, so a run that leans on the loose tiers is
visible rather than silent.

Sequence IDs are read from the genome FASTAs directly, in parallel. Nothing is
read from an ALASIGHT database, so this can run before one exists, or against
genomes indexed elsewhere - the only inputs are a list of genome paths and the
GTDB metadata.

Inputs:
  --fasta-list    a list of genome FASTA paths, one per line (.gz is fine)
  --metadata      bac120_metadata_r214.tsv, ar53_metadata_r214.tsv (.gz is fine);
                  pass both, or any number

Usage:
  gtdb_species_labels.py -f database/db_fasta_list.txt \\
      -m bac120_metadata_r214.tsv -m ar53_metadata_r214.tsv \\
      -o species_gtdb.tsv --unmatched unmatched_genomes.txt

Then rebuild:
  rm database/alasight_db_index.tsv
  alasight.py build-db -i genomes.txt -d database -tr timetree/ -s species_gtdb.tsv
"""

import argparse
import gzip
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# GCF_000009045.1 / GCA_000009045.1, with or without the version suffix.
ACCESSION = re.compile(r"GC[AF]_(\d{6,})(?:\.(\d+))?", re.IGNORECASE)
# GTDB placeholder epithets ('sp002506415') and placeholder genus codes
# ('UBA11063', 'CAG-495', 'Palsa-739', '2-02-FULL-50-16').
PLACEHOLDER = re.compile(r"^sp\d+$")
CODE_GENUS = re.compile(r"[\d-]")

TIERS = ("exact", "gca_gcf_swap", "digits_only")


def log(message):
    print(message, file=sys.stderr, flush=True)


def smart_open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def accession_keys(text):
    """
    (full accession, swapped-prefix accession, digits) for the first assembly
    accession in `text`, or None if there is none.

    GCA and GCF share their digits for a paired assembly, which is what makes
    the swap and digits tiers legitimate rather than merely convenient. Callers
    on the metadata side register only the full form, so the loosening happens
    at lookup time and the reported tier is the real one.
    """
    match = ACCESSION.search(str(text))
    if not match:
        return None
    digits, version = match.group(1), match.group(2)
    prefix = match.group(0)[:3].upper()          # GCA or GCF
    other = "GCF" if prefix == "GCA" else "GCA"
    versioned = f"{digits}.{version}" if version else digits
    return f"{prefix}_{versioned}", f"{other}_{versioned}", digits


class SpeciesTable:
    """
    Two lookups: one keyed by the full accession, one by digits alone.

    Keeping them apart is what lets a match report the tier it really used - a
    single merged table would satisfy an 'exact' lookup with a key that had been
    loosened at load time. Either lookup can collide, and a key mapping to more
    than one species is recorded as ambiguous rather than silently taking
    whichever row came last: two versions of an assembly can sit in different
    species, and that is precisely what the digits tier ignores.
    """

    def __init__(self):
        self.by_accession = {}
        self.by_digits = {}
        self.ambiguous = set()

    def _add(self, table, key, species):
        if key in self.ambiguous:
            return
        seen = table.get(key)
        if seen is None:
            table[key] = species
        elif seen != species:
            table.pop(key, None)
            self.ambiguous.add(key)

    def add(self, accession, species):
        keys = accession_keys(accession)
        if not keys:
            return
        full, _swap, digits = keys
        self._add(self.by_accession, full, species)
        self._add(self.by_digits, digits, species)

    def lookup(self, keys):
        """(species, tier) for the first tier that resolves, else (None, reason)."""
        full, swap, digits = keys
        hit_ambiguous = False
        for tier, table, key in (("exact", self.by_accession, full),
                                 ("gca_gcf_swap", self.by_accession, swap),
                                 ("digits_only", self.by_digits, digits)):
            if key in self.ambiguous:
                hit_ambiguous = True
                continue
            species = table.get(key)
            if species:
                return species, tier
        return None, "ambiguous" if hit_ambiguous else "absent"


def species_from_lineage(lineage):
    """
    The 's__' field of a GTDB lineage, without its prefix.

    Only the species rank is taken: alasight's taxon_tokens reads the first two
    tokens of whatever it is given, so a full 'd__Bacteria;p__...' string would
    resolve to 'bacteria' and match the wrong node.
    """
    for field in str(lineage).split(";"):
        field = field.strip()
        if field.startswith("s__"):
            name = field[3:].strip()
            return name or None
    return None


def read_metadata(paths):
    """Build the accession -> species table from one or more GTDB metadata TSVs."""
    table = SpeciesTable()
    genomes = 0
    for path in paths:
        log(f"  reading {path}...")
        with smart_open(path) as f:
            header = f.readline().rstrip("\n").split("\t")
            try:
                acc_col = header.index("accession")
                tax_col = header.index("gtdb_taxonomy")
            except ValueError:
                sys.exit(f"{path} has no 'accession' and 'gtdb_taxonomy' columns; "
                         f"is it a GTDB metadata file?")
            # Optional, and worth using when present: it gives the GenBank
            # accession for a genome GTDB lists under its RefSeq one.
            extra = [header.index(name) for name in
                     ("ncbi_genbank_assembly_accession", "ncbi_refseq_assembly_accession")
                     if name in header]

            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(acc_col, tax_col):
                    continue
                species = species_from_lineage(parts[tax_col])
                if not species:
                    continue
                genomes += 1
                for col in [acc_col] + extra:
                    if col < len(parts):
                        table.add(parts[col], species)
    log(f"  {genomes:,} metadata genome(s), {len(table.by_accession):,} accession key(s), "
        f"{len(table.by_digits):,} digit key(s), {len(table.ambiguous):,} ambiguous")
    return table


def _scan_one(job):
    """Worker: sequence IDs in one genome file. Module level so it can be pickled."""
    file_idx, path = job
    ids = []
    try:
        with smart_open(path) as handle:
            for line in handle:
                if line.startswith(">"):
                    ids.append(line[1:].split(None, 1)[0])
    except OSError as exc:
        return file_idx, [], str(exc)
    return file_idx, ids, None


def seq_ids_by_file(fasta_paths, threads):
    """
    file index -> [sequence ids], read from the genome FASTAs themselves.

    Deliberately independent of any ALASIGHT database: the only inputs this
    script needs are a list of genome paths and the GTDB metadata, so it can be
    run before a database exists, or for a database built elsewhere. Only the
    '>' lines are parsed, and gzip is handled transparently.
    """
    log(f"  scanning {len(fasta_paths):,} genome file(s) on {threads} worker(s)...")
    jobs = list(enumerate(fasta_paths))
    by_file, read_errors = {}, {}
    started = time.monotonic()
    step = max(len(jobs) // 20, 1)

    def note(file_idx, ids, err):
        by_file[file_idx] = ids
        if err:
            read_errors[file_idx] = err

    if threads > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(threads, len(jobs))) as pool:
            for done, (file_idx, ids, err) in enumerate(
                    pool.map(_scan_one, jobs, chunksize=16), 1):
                note(file_idx, ids, err)
                if done % step == 0 or done == len(jobs):
                    rate = done / max(time.monotonic() - started, 1e-6)
                    log(f"    {done:,}/{len(jobs):,} ({rate:,.0f} files/s)")
    else:
        for job in jobs:
            note(*_scan_one(job))

    total = sum(len(v) for v in by_file.values())
    log(f"  {total:,} sequence ID(s) across {len(by_file):,} file(s) "
        f"in {time.monotonic() - started:.0f}s")
    if read_errors:
        log(f"  WARNING: {len(read_errors):,} genome file(s) could not be read")
    return by_file, read_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--fasta-list", required=True,
                    help="db_fasta_list.txt, one genome FASTA path per line")
    ap.add_argument("-m", "--metadata", required=True, action="append",
                    help="GTDB metadata TSV; repeat for bacteria and archaea")
    ap.add_argument("-o", "--output", required=True,
                    help="species TSV to write, for build-db --species-file")
    ap.add_argument("--unmatched", help="write genomes with no species label here")
    ap.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 8,
                    help="workers for the FASTA scan (default: all cores)")
    args = ap.parse_args()

    fasta_paths = [line.strip() for line in
                   Path(args.fasta_list).read_text().splitlines() if line.strip()]
    if not fasta_paths:
        sys.exit(f"{args.fasta_list} lists no genome files.")
    log(f"Genome files: {len(fasta_paths):,}")

    log("Reading GTDB metadata...")
    table = read_metadata(args.metadata)

    log("Collecting sequence IDs...")
    by_file, read_errors = seq_ids_by_file(fasta_paths, args.threads)

    log("Joining...")
    tier_counts = Counter()
    fail_counts = Counter()
    unmatched, species_seen = [], Counter()
    rows = matched_seqs = unmatched_seqs = 0

    with open(args.output, "w") as out:
        out.write("# ALASIGHT species labels from GTDB metadata\n")
        out.write(f"# fasta_list: {Path(args.fasta_list).resolve()}\n")
        for path in args.metadata:
            out.write(f"# metadata: {Path(path).resolve()}\n")
        out.write("#\n")

        for file_idx, path in enumerate(fasta_paths):
            ids = by_file.get(file_idx, [])
            if file_idx in read_errors:
                fail_counts["could not be read"] += 1
                unmatched.append((path, f"unreadable: {read_errors[file_idx]}"))
                continue
            keys = accession_keys(Path(path).name)
            if keys is None:
                fail_counts["no accession in filename"] += 1
                unmatched.append((path, "no accession in filename"))
                unmatched_seqs += len(ids)
                continue
            species, tier = table.lookup(keys)
            if species is None:
                fail_counts[f"accession {tier} from metadata"] += 1
                unmatched.append((path, f"accession {tier}"))
                unmatched_seqs += len(ids)
                continue
            tier_counts[tier] += 1
            species_seen[species] += 1
            for seq_id in ids:
                out.write(f"{seq_id}\t{species}\n")
                rows += 1
            matched_seqs += len(ids)

    if args.unmatched:
        with open(args.unmatched, "w") as f:
            for path, reason in unmatched:
                f.write(f"{path}\t{reason}\n")

    total_files = len(fasta_paths)
    matched_files = sum(tier_counts.values())
    # GTDB placeholder epithets ('sp002506415') and placeholder genera
    # ('UBA11063', 'CAG-495', 'Palsa-739'). The two are worth separating:
    # alasight's leaf_match falls back to genus, so a placeholder epithet under
    # a real genus can still resolve, while a placeholder genus cannot.
    placeholder_species = [s for s in species_seen
                           if PLACEHOLDER.match(s.split(" ")[-1]) and " " in s]
    placeholder_genus = [s for s in placeholder_species
                         if CODE_GENUS.search(s.split(" ")[0])]

    print("\n" + "=" * 68)
    print("GENOMES")
    print("=" * 68)
    print(f"  genome files                {total_files:>12,}")
    print(f"  matched to a GTDB species   {matched_files:>12,}"
          f"   ({100.0 * matched_files / total_files:.1f}%)")
    for tier in TIERS:
        if tier_counts[tier]:
            print(f"    via {tier:<22}{tier_counts[tier]:>12,}")
    for reason, count in fail_counts.most_common():
        print(f"  unmatched: {reason:<18}{count:>12,}")

    print("\n" + "=" * 68)
    print("SEQUENCES")
    print("=" * 68)
    print(f"  labelled                    {matched_seqs:>12,}")
    print(f"  unlabelled                  {unmatched_seqs:>12,}")
    total_seqs = matched_seqs + unmatched_seqs
    if total_seqs:
        print(f"  coverage                    {100.0 * matched_seqs / total_seqs:>11.1f}%")

    print("\n" + "=" * 68)
    print("SPECIES")
    print("=" * 68)
    print(f"  distinct species            {len(species_seen):>12,}")
    n_sp = len(species_seen) or 1
    print(f"  placeholder epithet         {len(placeholder_species):>12,}"
          f"   ({100.0 * len(placeholder_species) / n_sp:.1f}%)")
    print("    valid GTDB species with no binomial name; most still resolve via")
    print("    the genus fallback, so this is an upper bound on unusable labels")
    print(f"  placeholder genus too       {len(placeholder_genus):>12,}"
          f"   ({100.0 * len(placeholder_genus) / n_sp:.1f}%)")
    print("    no genus for the tree to fall back on, so these are the ones")
    print("    expected to stay at leaf NA after a rebuild")
    print("=" * 68)
    print(f"\nWrote {rows:,} label(s) to {args.output}")
    if args.unmatched and unmatched:
        print(f"Wrote {len(unmatched):,} unmatched genome(s) to {args.unmatched}")


if __name__ == "__main__":
    main()
