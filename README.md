# Moocleus Genomics

**Breed Better. Know More.** A genomic prediction engine for dairy cattle embryos, powered by the 1000 Bull Genomes Project.

Moocleus is a data engineering pipeline that processes whole-genome sequence variation data from 1,832 bulls to identify causal mutations affecting dairy production traits like milk yield, fat percentage, and fertility.

## Features

- **ENA Integration** — Automatically resolves and downloads VCF files from the European Nucleotide Archive
- **Parallel Downloads** — Multi-threaded wget/aria2c with resume support and dry-run mode
- **Dual VCF Backend** — cyvcf2 (Linux, fast) or scikit-allel (Windows, cross-platform)
- **Biological Filtering** — Retains high-impact variants (stop-gain, frameshift) and SNPs near 6 major dairy genes
- **Region-Targeted Extraction** — Tabix-indexed queries extract only gene regions (seconds, not hours)
- **Parquet Output** — Compressed feature matrices ready for ML training

## Installation

```bash
# Clone and install
git clone https://github.com/your-username/moocleus-genomics.git
cd moocleus-genomics
pip install -e .

# Linux: install cyvcf2 for fast VCF parsing
pip install -e ".[linux]"

# Windows: install scikit-allel fallback
pip install -e ".[windows]"
```

## Usage

```bash
# Preview the full pipeline (no downloads)
moocleus run-all --dry-run

# Run step by step
moocleus parse-manifest                    # Parse ENA manifest
moocleus resolve-urls                      # Resolve download URLs
moocleus download -chr 14 --tool wget      # Download Chr14 (~29 GB)
moocleus extract -chr 14                   # Extract genotypes near target genes
moocleus filter-variants -chr 14           # Apply biological priority filter

# Full pipeline (downloads + processing)
moocleus run-all

# Process existing VCFs (skip download)
moocleus run-all --skip-download
```

## Data

- **Source**: [1000 Bull Genomes Project Run 8](https://www.ebi.ac.uk/ena/browser/view/PRJEB42783) (PRJEB42783)
- **Scale**: 1,832 samples, 84M+ SNPs across 29 autosomes + X
- **Species**: Bos taurus (dairy cattle focus)
- **Target chromosomes**: 6, 14, 20, 26 (containing 6 major dairy genes)

## Target Genes

| Gene | Chromosome | Trait | Reference |
|------|-----------|-------|-----------|
| DGAT1 | 14 | Milk fat percentage | Grisart et al. 2002 |
| ABCG2 | 6 | Milk yield & composition | Cohen-Zinder et al. 2005 |
| GHR | 20 | Growth & milk production | Blott et al. 2003 |
| CSN2 | 6 | Beta-casein (A1/A2) | Farrell et al. 2004 |
| SCD1 | 26 | Fatty acid composition | Taniguchi et al. 2004 |
| PLAG1 | 14 | Stature & body size | Karim et al. 2011 |

## Pipeline Architecture

```
filereport_analysis_PRJEB42783.tsv
  │
  ▼
ManifestParser ── filter Bos taurus + SEQUENCE_VARIATION
  │
  ▼
ENAResolver ── ENA REST API → per-chromosome FTP URLs
  │
  ▼
VCFDownloader ── parallel wget/aria2c with resume
  │
  ▼
VCFFeatureExtractor ── cyvcf2 or scikit-allel, region-targeted
  │
  ▼
BiologicalPriorityFilter ── high-impact + 50kb gene proximity
  │
  ▼
ParquetWriter ── compressed feature matrices for ML
```

## Output

The pipeline produces:
- `data/processed/chr{N}_{GENE}_filtered.parquet` — Per-gene genotype matrices (samples x SNPs)
- `data/processed/variant_catalog.parquet` — Summary of all retained variants

## Tech Stack

| Component | Technology |
|-----------|-----------|
| CLI | Click |
| Data | pandas, pyarrow |
| VCF Parsing | cyvcf2 / scikit-allel |
| Downloads | wget / aria2c |
| Config | PyYAML |
| Progress | tqdm |

## License

MIT
