# Moocleus Genomics

Bovine genomic data engineering pipeline for the 1000 Bull Genomes Project.

## Quick Start

```bash
pip install -e .
moocleus --help
moocleus run-all --dry-run
```

## Architecture

```
TSV Manifest → ManifestParser → ENAResolver → VCFDownloader → VCFFeatureExtractor → BiologicalPriorityFilter → ParquetWriter
```

## Package Layout

- `moocleus/config.py` — PipelineConfig + GeneRegion dataclasses (loaded from conf/pipeline.yaml)
- `moocleus/cli.py` — Click CLI entry point with 6 commands
- `moocleus/manifest/parser.py` — ENA TSV parser with species/type filtering
- `moocleus/resolver/ena_resolver.py` — ENA REST API URL resolver with fallback
- `moocleus/downloader/vcf_downloader.py` — Parallel wget/aria2c downloader with dry-run + resume
- `moocleus/extractor/vcf_parser.py` — cyvcf2-based VCF feature extraction (Linux)
- `moocleus/extractor/vcf_parser_fallback.py` — scikit-allel fallback (Windows)
- `moocleus/filters/bio_filter.py` — High-impact + gene-proximity variant filter
- `moocleus/io/parquet_writer.py` — Parquet output with embedded metadata

## Key Commands

```bash
moocleus parse-manifest              # Show filtered accessions
moocleus resolve-urls -chr 14        # Resolve FTP URLs
moocleus download --dry-run          # Preview downloads
moocleus download -chr 14            # Download Chr14 VCF (~29 GB)
moocleus extract -chr 14             # Extract genotypes for target genes
moocleus filter-variants -chr 14     # Apply biological filters
moocleus run-all --dry-run           # Full pipeline preview
moocleus run-all --skip-download     # Run extract+filter on existing VCFs
```

## Target Genes (ARS-UCD2.0)

| Gene  | Chr | Region            | Trait              |
|-------|-----|-------------------|--------------------|
| DGAT1 | 14  | 611,019-621,088   | Milk fat           |
| ABCG2 | 6   | 37.96M-38.02M     | Milk yield         |
| GHR   | 20  | 31.89M-32.20M     | Growth hormone     |
| CSN2  | 6   | 87.14M-87.15M     | Beta-casein        |
| SCD1  | 26  | 21.14M-21.15M     | Fatty acid         |
| PLAG1 | 14  | 24.97M-25.00M     | Stature            |

## Dependencies

Core: click, pandas, pyarrow, numpy, requests, tqdm, pyyaml
VCF (Linux): cyvcf2
VCF (Windows): scikit-allel
