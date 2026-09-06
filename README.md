# medival

ALASIGHT

**A**pproximate **L**ocal **A**lignment for **SIG**natures of **H**orizontal **T**ransfer.
A tool for detecting signatures of horizontal gene transfer, which might signal the presence of Mobile Genetic Elements (MGEs) using alamem and phylogenetic divergence analysis.
## Overview

Alasight finds signatures of horizontal gene transfer by using alamem to in parallel find all hits across a GTDB database, applying divergence filtering, and then running an overlap-divergence filter to identify regions supported by alignments to distantly related species. It uses the TimeTree of Life to calculate divergence times and skani for average nucleotide identity (ANI) lookups. We designed this tool to use HGT to find novel MGEs that do not look like reference MGE databases, by doing a string similarity search across entire bacterial genome databases.

## Installation
Install medival:
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
python3 medival.py -h

# build database (assumes you've already created gtdb_list.txt with correct absolute paths
mkdir database
mkdir tree
./medival.py build-db -i references-local/gtdb_list.txt -d database -tr tree -n references-local/TimeTree_v5_Final.nwk --species-file references-local/all_gtdb_id_and_kraken_species.txt -t 64

# command line with only required arguments
./medival.py run -d database -q input.fasta -o out_dir -t 64

```
### Parameters

#### Required Arguments
| Parameter | Description |
|-----------|-------------|
| `-q, --query`| Path to the query FASTA file |
| `-o, --output`| Name of your output directory 
| `-d, --database`| Path to the medival database directory| |
| `-tr, --tree`| Path to the phylogenetic tree directory |

#### Optional Arguments
| Parameter | Default | Description |
|-----------|---------|-------------|
| `-t, --threads` | 1 | Number of threads. **Highly recommended to increase.** |
| `-s, --species` | auto-detect | Species name for all sequences in the input FASTA (replace spaces with `_`). Cannot be used with `--speciesFile`. |
| `--speciesFile` | auto-detect | Tab-separated file assigning a species to each sequence ID. Cannot be used with `-s`. |
| `-minScore` | 30 | Minimum BLAT alignment score. |
| `-minIdentity` | 90 | Minimum percent identity: `(matches / (Q_end − Q_start)) × 100`. |
| `--size-filter` | 150 | Discard final regions smaller than this many bp. |
| `--cluster-size` | 0 | Merge final regions within this many bp of each other. |


### Example Commands

#### Basic command
```bash
python3 medival.py \
  -q my_sequences.fasta \
  -o results_dir \
  -d /path/to/medival_gtdb_db/ \
  -tr /path/to/divergence_tree/ \
  -k /path/to/kraken2_db/ \
  -t 20
```

#### With stricter filtering and custom thresholds
```bash
python3 medival.py \
  -q my_sequences.fasta \
  -o results_dir \
  -d /path/to/medival_gtdb_db/ \
  -tr /path/to/divergence_tree/ \
  -k /path/to/kraken2_db/ \
  -t 20 \
  -minIdentity 95 \
  --size-filter 150
```
## Filters
Filters are described in further detail, and their processes are illustrated in our paper (add cite).
### Divergence Filtering
- Divergence filtering is the main method for detecting MGEs and produces the file ```{output_name}_first_div_output.tsv```.
- By examining the species of the query genome and the genome it aligned to, we use the TimeTree of Life to calculate the divergence time between the two species. If the species diverged over 1 million years ago (```divergence >= 1 MYA```), the alignment is retained.
- This filter is effective at identifying horizontal gene transfer events because MGEs transferred between distantly related species will show high sequence similarity despite ancient species divergence.
- For detailed information on how this filter detects MGEs, please refer to our paper: (citation tba).
### Overlap-Divergence Filtering (always runs)
This filter produces ```{output_name}_overlap_div.tsv```:
- Finds pairs of BLAT hits that overlap on the query sequence and whose reference sequences are divergently distant from each other (≥ 1 MYA), or have ANI < 95% when divergence is unknown.
- Removes false positives caused by self-alignments or hits from closely related organisms.
- ANI between reference sequence pairs is looked up in a pre-computed all-vs-all skani triangle (```skani_triangle_ani95.pkl```), making this filter much faster than per-hit skani calls.

### Size and Cluster Filtering (always runs)
Final regions are built from the overlap-div output:
- Intervals within `--cluster-size` bp of each other are merged (default: 0 bp)
- Regions smaller than `--size-filter` bp are discarded (default: 250 bp)
- Produces ```{output_name}_final_regions.tsv``` and ```{output_name}_final_regions_summary.tsv```

### Output Files
```
output_directory/
├── output_name_blat_results.tsv           # Raw BLAT alignments (all chunks combined)
├── output_name_first_div_output.tsv       # Divergence-filtered results
├── output_name_overlap_div.tsv            # Overlap + divergence filtered results
├── output_name_final_regions.tsv          # Final MGE regions (size + cluster filtered)
├── output_name_final_regions_summary.tsv  # Per-region summary statistics
├── skani_ani_dict_*.pkl                   # Cached skani query-vs-reference ANI results
├── species_*.tsv                          # Auto-detected species assignments
└── intermediate_{output_name}_files/      # Per-chunk intermediate files
    ├── chunk_0_{name}/                    # Raw .psl files from BLAT
    │   ├── chunk_0_{name}_part_0.psl
    │   ...
    │   └── chunk_0_{name}_part_136.psl
    ├── chunk_0_{name}_blat_output.tsv     # Combined BLAT output for this chunk
    ├── chunk_0_{name}_first_div_output.tsv
    ├── chunk_0_{name}_overlap_div.tsv
    ...
```
The `intermediate_{output_name}_files/` directory holds per-chunk working files. The GTDB-BLAT database is split into 136 parts for parallel processing; each chunk of the query is run against all 136 parts, and the results are combined before filtering. Once you are satisfied with your results, you can safely **delete the intermediate directory** to free disk space.

All output files begin with a `#`-prefixed configuration header recording the parameters and timestamp of the run.

**`final_regions.tsv`** contains one row per final MGE region. Reference metadata columns (T name, Divergence Time, etc.) are merged using `|` as a row delimiter and `,` within a row — each `|`-delimited token represents one contributing overlap-div hit (which itself is a pair of reference sequences). To recover individual contributing hits, split on `|`.

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
Important: The program is designed to resume from interruptions by checking for existing files. If a run is stopped prematurely, it will restart from where it left off. Avoid creating files with names that could overlap with medival's output to prevent conflicts.

## Database Setup

#### Kraken and Kraken Database
We provide the kraken database ```kraken2_custom_db```. If you want to recreate it locally, do the following:
```bash
mkdir -p kraken2_custom_db

kraken2-build --download-taxonomy --db kraken2_custom_db

kraken2-build --download-library archaea --db kraken2_custom_db
kraken2-build --download-library bacteria --db kraken2_custom_db
kraken2-build --download-library plasmid --db kraken2_custom_db
kraken2-build --download-library viral --db kraken2_custom_db
kraken2-build --download-library fungi --db kraken2_custom_db
kraken2-build --download-library protozoa --db kraken2_custom_db
kraken2-build --download-library nt --db kraken2_custom_db

kraken2-build --build --db kraken2_custom_db
kraken2-build --clean --db kraken2_custom_db
```

#### Pre-built GTDB Database
We provide a ready-to-use database built from the Genome Taxonomy Database (GTDB):
- **BLAT database:** ```medival_gtdb_db/blat_2bit_db/```
- **Index:** ```medival_gtdb_db/medival_db_index.pkl```
- **skani sketches:** ```medival_gtdb_db/skani_sketch_db/```
- **skani triangle:** ```medival_gtdb_db/skani_triangle_ani95.pkl```

### Creating a Custom Database
If you want to use your own genome collection, follow these steps:
#### Step 1: Build the BLAT and skani databases
```
python3 make_medival_db.py [fasta_file] [output_name] [size_in_bil_bp] [threads (optional, default 1)]
```
For example:
```
python3 make_medival_db.py gtdb.fa medival_gtdb_db 2
```
This script will:
- Split the input FASTA into chunks of the specified size (in billions of bp)
- Convert each chunk to 2bit format and generate `.ooc` files
- Run `skani sketch` on all sequences to create the sketch database
- Run `skani triangle` to compute all-vs-all ANI ≥ 95% pairs and save as `skani_triangle_ani95.pkl`
#### Step 2: Extract Species Information
Classify sequences using Kraken2:
```
python3 extract_species_from_kraken.py [fasta file] [kraken_db]
```
##### Outputs:
- ```{fasta_file}_kraken_output.txt```: Kraken's detailed classification output
- ```{fasta_file}_seq_species.txt```: Tab-separated file with sequence ID and species:
```
seq_id1  species1
seq_id2  species2
...  ...
```
You can manually edit species assignments in the ```{fasta_file}_seq_species.txt``` file if needed.
#### Step 3: Build the Database Index
Create a hash table index linking sequence IDs to their species, length, and leaf name in the phylogenetic tree:
```
python3 build_database_index.py [medival_database_from_make_medival_db.py] [{fasta_file}_seq_species.txt] [divergence_tree]
```
This creates an index where ```index[seq_id] = (species, length, leaf name in tree)```.
This index is crucial to the pipeline's speed.

## Required Files Summary
1. **BLAT Database:** Directory containing .2bit and .ooc files
2. **Index File:** .pkl file mapping sequence IDs to species, length, and leaf name in the phylogeny tree.
3. **TimeTree Directory:** Phylogenetic tree with relevant species, preprocessed,
4. **Kraken Database:** For taxonomic classification

## Phylogenetic Tree
We use the Time Tree of Life to calculate divergence times between species. From the original .nwk TimeTree file, we preprocessed the tree (as shown in this [Google Colab notebook](https://colab.research.google.com/drive/14xnM2kPtqHvnQi7cvWDLJedbZ0uWtqMy?usp=sharing)
) to allow finding the closest common ancestor in O(1) time. If a new .nwk file from the Time Tree becomes available, this notebook can be used to generate updated indexes and preprocess the tree for efficient queries. (cite timetree)

## Performance Tips
1. **Use many threads:** `-t 46` or higher significantly speeds up BLAT and skani steps.
2. **Chunk size:** The default 100 kb works well. Increasing it slows BLAT substantially.
3. **Re-runs are fast:** The skani ANI dict and species file are cached in the output directory; re-running with different `--size-filter` or `--cluster-size` values reuses all intermediate files and completes quickly.
4. **Disk space:** Intermediate files can be large for long sequences. Once you are satisfied with your results, delete the `intermediate_{output_name}_files/` directory to free disk space.
5. **Resume feature:** Take advantage of the automatic resume capability — re-running the same command after an interruption picks up from where it left off.

## Workflow
**Species detection:** Determines query species via Kraken2 (unless provided by `-s` or `--speciesFile`). Results are saved and reused on subsequent runs.

**skani search:** Queries all input sequences against the skani sketch database to identify reference sequences with ≥ 95% ANI. Results are cached and reused.

**Chunking:** Splits sequences longer than `--chunk` bp into manageable pieces.

**Parallel BLAT:** Each chunk is run against all 136 BLAT database parts in parallel.

**Divergence filter:** Retains hits where query and reference species diverged ≥ 1 MYA. Uses cached skani ANI as a fallback when divergence is unknown.

**Overlap-divergence filter:** Identifies overlapping hit pairs whose reference sequences are from divergent lineages. Always runs.

**Size + cluster filter:** Merges nearby regions and removes small ones to produce the final MGE calls.


## Citation

If you use medival in your research, please cite:
[Add later]

## Support

For questions and support, please submit an issue ticket on the GitHub repository.
