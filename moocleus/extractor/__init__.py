"""VCF feature extraction with platform-aware backend selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moocleus.extractor.vcf_parser import VCFFeatureExtractor
    from moocleus.extractor.vcf_parser_fallback import VCFFeatureExtractorFallback


def get_extractor(
    backend: str = "cyvcf2", chunk_size: int = 10_000
) -> VCFFeatureExtractor | VCFFeatureExtractorFallback:
    """Factory function to get the appropriate VCF parser backend."""
    if backend == "cyvcf2":
        try:
            from moocleus.extractor.vcf_parser import VCFFeatureExtractor
            return VCFFeatureExtractor(chunk_size=chunk_size)
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "cyvcf2 not available, falling back to scikit-allel"
            )
            from moocleus.extractor.vcf_parser_fallback import VCFFeatureExtractorFallback
            return VCFFeatureExtractorFallback(chunk_size=chunk_size)
    elif backend == "scikit-allel":
        from moocleus.extractor.vcf_parser_fallback import VCFFeatureExtractorFallback
        return VCFFeatureExtractorFallback(chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown VCF backend: {backend}. Use 'cyvcf2' or 'scikit-allel'.")
