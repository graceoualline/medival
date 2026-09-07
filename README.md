# alasight

ALASIGHT

**A**pproximate **L**ocal **A**lignment for **SIG**natures of **H**orizontal **T**ransfer.
A tool for detecting signatures of horizontal gene transfer, which might signal the presence of Mobile Genetic Elements (MGEs) using alamem and phylogenetic divergence analysis.
## Overview

Alasight finds signatures of horizontal gene transfer by using alamem to in parallel find all hits across a GTDB database, applying divergence filtering, and then running an overlap-divergence filter to identify regions supported by alignments to distantly related species. It uses the TimeTree of Life to calculate divergence times and skani for average nucleotide identity (ANI) lookups. We designed this tool to use HGT to find novel MGEs that do not look like reference MGE databases, by doing a string similarity search across entire bacterial genome databases.

## Installation
Install alasight:
```bash
git clone https://github.com/graceoualline/medival.git
cd medival
```

We have included in the references folder a species conversion table for GTDB r214 reference genomes to NCBI annotations, though you will have to unxz it. This allows us to map those hits onto the TimeTree of Life (also included) for divergence computations.

You will need to have the [GTDB r214 reference database](https://data.gtdb.ecogenomic.org/releases/release214/214.1/genomic_files_reps/gtdb_genomes_reps_r214.tar.gz) untarred somewhere (the individual fastas can stay gzipped or be decompressed, as you will).
Then generate a file with absolute paths to all of the genomes using
```bash
cp -a references-compressed references-local
cd references-local/
find [path/to/genomes_reps_r214] > gtdb_list.txt
unxz all_gtdb_id_and_kraken_species.txt.xz

```

You should have the following files:
```
└── references-local
    ├── all_gtdb_id_and_kraken_species.txt
    └── TimeTree_v5_Final.nwk
```

Also, you'll need to install required Python packages
```
### Required Python Packages
```bash
pip install biopython tqdm pyyaml matplotlib
```
### Prerequisites
Please ensure you have the following tools installed:
- Python 3.7+
- skani: https://github.com/bluenote-1577/skani
- alamem: https://github.com/yunwilliamyu/alamem

## Usage

### Quick Start

```bash
# To see all input parameters
python3 alasight.py -h

# build database (assumes you've already created gtdb_list.txt with correct absolute paths
mkdir database
mkdir tree
chmod +x alasight.py
./alasight.py build-db -i references-local/gtdb_list.txt -d database -tr tree -n references-local/TimeTree_v5_Final.nwk --species-file references-local/all_gtdb_id_and_kraken_species.txt -t 64

# command line with only required arguments
./alasight.py run -d database -q input.fasta -o out_dir -t 64

```
### Parameters

#### Required Arguments
| Parameter | Description |
|-----------|-------------|
| `-q, --query`| Path to the query FASTA file |
| `-o, --output`| Name of your output directory 
| `-d, --database`| Path to the alasight database directory| |
| `-tr, --tree`| Path to the phylogenetic tree directory |

#### Optional Arguments
| Parameter | Default | Description |
|-----------|---------|-------------|
| `-t, --threads` | 1 | Number of threads. **Highly recommended to increase.** |
| `-s, --species-file` | auto-detect | Tab-separated file assigning a species to each sequence ID. Cannot be used with `-s`. |
| `--species` | auto-detect | Species name for all sequences in the input FASTA (replace spaces with `_`). Output metadata only, doesn't affect hits |
| `--min-len` | 40 | Minimum length of alamem hit. |
| `--min-ani` | 90 | Minimum percent identity: `(matches / (Q_end − Q_start)) × 100`. |
| `--size-filter` | 150 | Discard final regions smaller than this many bp. |
| `--cluster-size` | 0 | Merge final regions within this many bp of each other. |


## Filters
Filters are described in further detail, and their processes are illustrated in our paper (add cite).
### ANI/Divergence Filtering
- ANI divergence filtering is the initial method for detecting HGT and produces the file ```{output_name}_first_div_output.tsv```. 
- We examine the ANI between the query genome and the genome it aligned to, and only retain if the ANI <= 95%, a typical species boundary.
- This filter is effective at identifying horizontal gene transfer events because MGEs transferred between distantly related species will show high sequence similarity despite ancient species divergence.
- For detailed information on how this filter detects horizontal gene transfer, please refer to our paper: (citation tba).
### Overlap-Divergence Filtering (always runs)
This filter produces ```{output_name}_overlap_div.tsv```:
- Finds pairs of hits that overlap on the query sequence and whose reference sequences are divergently distant from each other (≥ 1 MYA), or have ANI < 95% when divergence time is unknown.
- Removes false positives caused by self-alignments or hits from closely related organisms.
- ANI between reference sequence pairs is looked up via skani triangle on all the hit genomes.

### Size and Cluster Filtering (always runs)
Final regions are built from the overlap-div output:
- Intervals within `--cluster-size` bp of each other are merged (default: 0 bp)
- Regions smaller than `--size-filter` bp are discarded (default: 150 bp)
- Produces ```{output_name}_final_regions.tsv``` and ```{output_name}_final_regions_summary.tsv```

### Output Files
```
output_directory/
├── output_name_alamem_results.tsv         # Raw BLAT alignments
├── output_name_first_div_output.tsv       # ANI Divergence-filtered results
├── output_name_overlap_div.tsv            # Overlap + divergence (time and ANI) filtered results
├── output_name_final_regions.tsv          # Final MGE regions (size + cluster filtered)
├── output_name_final_regions_summary.tsv  # Per-region summary statistics
├── skani_ani.tsv                          # Cached skani query-vs-reference ANI results
```

All output files begin with a `#`-prefixed configuration header recording the parameters and timestamp of the run.

**`final_regions.tsv`** contains one row per final HGT region. Reference metadata columns (T name, Divergence Time, etc.) are merged using `|` as a row delimiter and `,` within a row — each `|`-delimited token represents one contributing overlap-div hit (which itself is a pair of reference sequences). To recover individual contributing hits, split on `|`.

**`final_regions_summary.tsv`** contains one row per region with the following columns:

| Column | Description |
|--------|-------------|
| `Q name` | Query sequence identifier |
| `Q size` | Full length of the query sequence (bp) |
| `Q start` / `Q end` | Coordinates of the final region on the query |
| `Query Species` | Species of the query sequence |
| `Num Regions` | Number of overlap-div hits that were merged into this region |
| `Num Unique Species` | Number of distinct reference species (by tree leaf name) that contributed hits |
| `Avg Divergence Time` | Average divergence time (MYA) across all contributing hits with a known divergence |

### Resume Functionality
Important: The program is designed to resume from interruptions by checking for existing files. If a run is stopped prematurely, it will restart from where it left off. Avoid creating files with names that could overlap with alasight's output to prevent conflicts.

## Database Setup

### Creating a Custom Database
If you want to use your own genome collection, you only need the following files:
- sequence_id_to_species_id.txt (in TSV format)
- species_tree.nwk (in Newark file format)
- fasta_list.txt (pointing to all the fasta files to be indexed)
```bash
./alasight.py build-db -i fasta_list.txt -d database -tr tree -n species_tree.nwk -s sequence_id_to_species_id.txt -t 64
```
## Phylogenetic Tree
We use the Time Tree of Life to calculate divergence times between species. If a new .nwk file from the Time Tree becomes available, you can use the `build-tree` option directly. (normally it is called indirectly from `build-db`).

## Performance Tips
1. **Use many threads:** `-t 64` or higher significantly speeds up alamem and skani steps.
2. **Resume feature:** Take advantage of the automatic resume capability for the build-db — re-running the same command after an interruption picks up from where it left off.

## Workflow
**skani search:** Queries all input sequences against the skani sketch database to identify reference sequences with ≥ 95% ANI.

**Divergence filter:** Retains hits where query and reference species have <=95% ANI.

**Overlap-divergence filter:** Identifies overlapping hit pairs whose reference sequences are from divergent lineages (<=95% ANI or >= 1 MYA). Always runs.

**Size + cluster filter:** Merges nearby regions and removes small ones to produce the final MGE calls.


## Citation

If you use alasight in your research, please cite:
[Add later]

## Support

For questions and support, please submit an issue ticket on the GitHub repository.
