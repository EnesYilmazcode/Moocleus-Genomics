"""Cross-platform VCF parser using scikit-allel (works on Windows).

On Windows, scikit-allel's region= parameter requires the external 'tabix'
binary which is not typically available.  We use full-file reads with fast
inline CHROM detection and in-memory filtering instead.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


class VCFFeatureExtractorFallback:
    """scikit-allel based VCF parser for Windows (no cyvcf2 dependency).

    Reads the full VCF once, detects CHROM naming inline, and filters
    to the requested region in memory.
    """

    def __init__(self, chunk_size: int = 10_000) -> None:
        self.chunk_size = chunk_size
        try:
            import allel  # noqa: F401
        except ImportError:
            raise ImportError(
                "scikit-allel is not installed. Install with: pip install scikit-allel"
            )

    def _detect_chrom_name(self, vcf_path: Path) -> str:
        """Read the first data line of a VCF to detect CHROM naming.

        This is very fast (~milliseconds) since it only decompresses
        until the first non-header line is found.
        """
        try:
            with gzip.open(str(vcf_path), "rt") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    chrom = line.split("\t", 1)[0]
                    logger.info("Detected VCF CHROM name: '%s'", chrom)
                    return chrom
        except Exception as e:
            logger.warning("Could not detect CHROM name: %s", e)
        return ""

    def extract_region(
        self, vcf_path: Path, region: str
    ) -> "ChromosomeFeatureMatrix":
        """Extract genotype matrix for a region using scikit-allel.

        Reads the full file once, then filters to the target region in memory.
        For a 29 GB file this takes ~15-20 minutes on the first read.
        """
        import allel
        from moocleus.extractor.vcf_parser import ChromosomeFeatureMatrix

        # Parse the region: expect "chrom:start-end"
        if ":" in region:
            chrom, coords = region.split(":", 1)
        else:
            chrom = region
            coords = None

        logger.info("Extracting region %s from %s (scikit-allel)", region, vcf_path.name)

        # Fast CHROM detection (~milliseconds, reads one gzip block)
        vcf_chrom = self._detect_chrom_name(vcf_path)
        if not vcf_chrom:
            vcf_chrom = chrom

        # Full-file read (scikit-allel's region= param requires external tabix
        # binary which is not available on Windows, so we read + filter manually)
        logger.info(
            "Reading full VCF into memory (this takes ~15-20 min for large files)..."
        )
        callset = allel.read_vcf(
            str(vcf_path),
            fields=[
                "variants/CHROM", "variants/POS", "variants/REF",
                "variants/ALT", "calldata/GT", "samples",
            ],
        )

        if callset is None:
            logger.warning("No variants found in %s", vcf_path.name)
            return ChromosomeFeatureMatrix(
                chromosome=chrom,
                sample_ids=[],
                positions=[],
                ref_alleles=[],
                alt_alleles=[],
                genotype_matrix=np.empty((0, 0), dtype=np.int8),
                info_annotations=[],
            )

        logger.info(
            "Loaded %d variants for %d samples",
            len(callset["variants/POS"]), len(callset["samples"]),
        )

        # Filter to the target region in memory
        if coords:
            query_region = f"{vcf_chrom}:{coords}"
            logger.info("Filtering to region %s in memory...", query_region)
            callset = self._filter_callset_by_region(callset, query_region)

        if callset is None:
            logger.warning("No variants found in region %s", region)
            return ChromosomeFeatureMatrix(
                chromosome=chrom,
                sample_ids=[],
                positions=[],
                ref_alleles=[],
                alt_alleles=[],
                genotype_matrix=np.empty((0, 0), dtype=np.int8),
                info_annotations=[],
            )

        gt = callset["calldata/GT"]  # (n_variants, n_samples, ploidy)
        dosage = gt.sum(axis=2)  # (n_variants, n_samples)
        missing_mask = (gt < 0).any(axis=2)
        dosage[missing_mask] = -1

        n_variants = dosage.shape[0]
        logger.info(
            "Extracted %d variants for %d samples",
            n_variants, len(callset["samples"]),
        )

        return ChromosomeFeatureMatrix(
            chromosome=chrom,
            sample_ids=list(callset["samples"]),
            positions=callset["variants/POS"].tolist(),
            ref_alleles=callset["variants/REF"].tolist(),
            alt_alleles=[",".join(a) for a in callset["variants/ALT"]],
            genotype_matrix=dosage.T.astype(np.int8),  # (n_samples, n_variants)
            info_annotations=[{} for _ in range(n_variants)],
        )

    def extract_full_chromosome(
        self, vcf_path: Path, chromosome: str
    ) -> Iterator["ChromosomeFeatureMatrix"]:
        """Read entire chromosome and yield in chunks."""
        import allel
        from moocleus.extractor.vcf_parser import ChromosomeFeatureMatrix

        logger.info("Reading chromosome %s from %s (scikit-allel)", chromosome, vcf_path.name)

        callset = allel.read_vcf(
            str(vcf_path),
            fields=[
                "variants/POS", "variants/REF", "variants/ALT",
                "calldata/GT", "samples",
            ],
        )

        if callset is None:
            logger.warning("No data for chromosome %s", chromosome)
            return

        gt = callset["calldata/GT"]
        dosage = gt.sum(axis=2)
        missing_mask = (gt < 0).any(axis=2)
        dosage[missing_mask] = -1

        positions = callset["variants/POS"]
        refs = callset["variants/REF"]
        alt_array = callset["variants/ALT"]
        samples = list(callset["samples"])
        n_total = len(positions)

        for start in range(0, n_total, self.chunk_size):
            end = min(start + self.chunk_size, n_total)
            chunk_dosage = dosage[start:end]

            yield ChromosomeFeatureMatrix(
                chromosome=chromosome,
                sample_ids=samples,
                positions=positions[start:end].tolist(),
                ref_alleles=refs[start:end].tolist(),
                alt_alleles=[",".join(a) for a in alt_array[start:end]],
                genotype_matrix=chunk_dosage.T.astype(np.int8),
                info_annotations=[{} for _ in range(end - start)],
            )

    def _filter_callset_by_region(self, callset: dict, region: str) -> dict | None:
        """Manually filter a callset to a specific region."""
        if ":" not in region:
            return callset

        chrom_str, coords = region.split(":", 1)
        start, end = (int(x) for x in coords.split("-"))

        positions = callset["variants/POS"]
        chroms = callset["variants/CHROM"]

        # Match chromosome: exact match first, then try bare (strip prefixes)
        bare_target = chrom_str.replace("Chr", "").replace("chr", "")
        chrom_mask = np.zeros(len(chroms), dtype=bool)
        for i, c in enumerate(chroms):
            if c == chrom_str:
                chrom_mask[i] = True
            else:
                bare_c = c.replace("Chr", "").replace("chr", "")
                if bare_c == bare_target:
                    chrom_mask[i] = True

        pos_mask = (positions >= start) & (positions <= end)
        mask = chrom_mask & pos_mask

        if not mask.any():
            return None

        return {
            "variants/CHROM": callset["variants/CHROM"][mask],
            "variants/POS": callset["variants/POS"][mask],
            "variants/REF": callset["variants/REF"][mask],
            "variants/ALT": callset["variants/ALT"][mask],
            "calldata/GT": callset["calldata/GT"][mask],
            "samples": callset["samples"],
        }
