"""Run the full data pipeline end-to-end."""

import importlib
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline():
    """Execute all pipeline steps in order."""
    print("=" * 60)
    print("MOOCLEUS GENOMICS — Data Pipeline")
    print("=" * 60)

    steps = [
        "ml.pipeline.01_build_snp_panel",
        "ml.pipeline.02_build_effect_weights",
        "ml.pipeline.03_simulate_genotypes",
        "ml.pipeline.04_simulate_phenotypes",
        "ml.pipeline.05_simulate_embryos",
        "ml.pipeline.06_export_demo_data",
    ]

    for step_module in steps:
        mod = importlib.import_module(step_module)
        mod.main()
        print()

    print("=" * 60)
    print("Pipeline complete! Demo data ready at app/demo_data.json")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
