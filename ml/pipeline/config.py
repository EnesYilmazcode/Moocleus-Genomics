"""Shared configuration for the ML pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PANEL_PATH = PROCESSED_DIR / "moocleus_panel.json"
WEIGHTS_DIR = PROCESSED_DIR / "trait_weights"
SIMULATED_DIR = PROCESSED_DIR / "simulated"
ARTIFACTS_DIR = ML_DIR / "artifacts"

# Bovine genome: 29 autosomes + X
CHROMOSOMES = list(range(1, 30)) + ["X"]

# Approximate chromosome lengths in base pairs (ARS-UCD2.0)
CHR_LENGTHS = {
    1: 158_534_110, 2: 136_231_102, 3: 121_005_158, 4: 120_000_601,
    5: 120_089_316, 6: 119_458_736, 7: 112_638_659, 8: 113_384_836,
    9: 105_708_250, 10: 104_305_016, 11: 107_310_763, 12: 87_216_183,
    13: 83_472_345, 14: 82_403_003, 15: 85_007_780, 16: 81_013_979,
    17: 73_167_244, 18: 65_820_629, 19: 63_449_741, 20: 71_974_595,
    21: 69_862_954, 22: 60_773_035, 23: 52_498_615, 24: 62_317_253,
    25: 42_350_435, 26: 51_992_305, 27: 45_612_108, 28: 45_940_150,
    29: 51_098_607, "X": 139_009_144,
}

# Trait definitions with published heritabilities and genetic parameters
TRAITS = {
    "milk_yield": {
        "name": "Milk Yield",
        "unit": "kg/lactation",
        "h2": 0.28,
        "mean": 10_500,
        "std": 1_800,
        "n_causal_snps": 500,
        "description": "305-day mature-equivalent milk yield",
    },
    "fat_pct": {
        "name": "Fat %",
        "unit": "%",
        "h2": 0.26,
        "mean": 3.7,
        "std": 0.45,
        "n_causal_snps": 300,
        "description": "Milk fat percentage",
    },
    "protein_pct": {
        "name": "Protein %",
        "unit": "%",
        "h2": 0.23,
        "mean": 3.2,
        "std": 0.25,
        "n_causal_snps": 250,
        "description": "Milk protein percentage",
    },
    "fertility": {
        "name": "Fertility",
        "unit": "index",
        "h2": 0.04,
        "mean": 0.0,
        "std": 1.0,
        "n_causal_snps": 200,
        "description": "Daughter pregnancy rate (standardized)",
    },
    "scs": {
        "name": "Somatic Cell Score",
        "unit": "log2(SCC/100) + 3",
        "h2": 0.12,
        "mean": 2.8,
        "std": 0.6,
        "n_causal_snps": 150,
        "description": "Indicator of udder health (lower is better)",
    },
    "longevity": {
        "name": "Longevity",
        "unit": "months",
        "h2": 0.05,
        "mean": 0.0,
        "std": 1.0,
        "n_causal_snps": 180,
        "description": "Productive life (standardized)",
    },
}

# Composite index weights (approximate USDA Net Merit $ relative emphasis)
COMPOSITE_WEIGHTS = {
    "milk_yield": 0.25,
    "fat_pct": 0.20,
    "protein_pct": 0.20,
    "fertility": 0.15,
    "scs": -0.10,  # Negative because lower SCS is better
    "longevity": 0.10,
}

# Notable bovine genes with known large effects
NOTABLE_GENES = {
    "DGAT1": {
        "chr": 14, "pos": 1_801_116, "gene": "DGAT1",
        "variant": "K232A", "rsid": "rs109326954",
        "effect": "Major effect on fat content (+0.14% fat per copy of K allele)",
        "traits": ["fat_pct", "milk_yield"],
        "maf": 0.36,
    },
    "ABCG2": {
        "chr": 6, "pos": 38_027_010, "gene": "ABCG2",
        "variant": "Y581S", "rsid": "rs43702337",
        "effect": "Affects milk yield and composition",
        "traits": ["milk_yield", "fat_pct", "protein_pct"],
        "maf": 0.12,
    },
    "GHR": {
        "chr": 20, "pos": 31_909_478, "gene": "GHR",
        "variant": "F279Y", "rsid": "rs385640152",
        "effect": "Growth hormone receptor — impacts milk yield",
        "traits": ["milk_yield"],
        "maf": 0.18,
    },
    "CSN2": {
        "chr": 6, "pos": 87_181_619, "gene": "CSN2",
        "variant": "A1/A2", "rsid": "rs43703011",
        "effect": "Beta-casein variant affecting protein composition",
        "traits": ["protein_pct"],
        "maf": 0.42,
    },
    "SCD1": {
        "chr": 26, "pos": 21_144_705, "gene": "SCD1",
        "variant": "A293V", "rsid": "rs41255693",
        "effect": "Stearoyl-CoA desaturase — fatty acid composition",
        "traits": ["fat_pct"],
        "maf": 0.28,
    },
    "PLAG1": {
        "chr": 14, "pos": 25_015_640, "gene": "PLAG1",
        "variant": "regulatory", "rsid": "rs109815800",
        "effect": "Pleiotropic developmental gene — stature and growth",
        "traits": ["milk_yield", "longevity"],
        "maf": 0.22,
    },
}

# Known recessive conditions for carrier status
RECESSIVE_CONDITIONS = {
    "BLAD": {
        "name": "Bovine Leukocyte Adhesion Deficiency",
        "chr": 1, "gene": "ITGB2", "carrier_freq": 0.07,
    },
    "CVM": {
        "name": "Complex Vertebral Malformation",
        "chr": 3, "gene": "SLC35A3", "carrier_freq": 0.13,
    },
    "DUMPS": {
        "name": "Deficiency of Uridine Monophosphate Synthase",
        "chr": 1, "gene": "UMPS", "carrier_freq": 0.02,
    },
    "Brachyspina": {
        "name": "Brachyspina Syndrome",
        "chr": 21, "gene": "FANCI", "carrier_freq": 0.08,
    },
}

# Simulation parameters
N_HERD = 200
N_SNPS = 15_000
N_MATINGS = 5
EMBRYOS_PER_MATING = 10
RANDOM_SEED = 42

# Embryo names (cow-themed)
EMBRYO_NAMES = [
    "Daisy", "Buttercup", "Clover", "Magnolia", "Primrose",
    "Rosie", "Hazel", "Poppy", "Willow", "Ivy",
    "Fern", "Violet", "Bluebell", "Marigold", "Juniper",
    "Luna", "Stella", "Aurora", "Meadow", "Sage",
    "Pearl", "Ruby", "Amber", "Coral", "Jade",
    "Maple", "Birch", "Cedar", "Aspen", "Holly",
    "Opal", "Flora", "Iris", "Lily", "Heather",
    "Olive", "Ginger", "Honey", "Cinnamon", "Nutmeg",
    "Pepper", "Saffron", "Clementine", "Blossom", "Petal",
    "Truffle", "Mocha", "Latte", "Cocoa", "Caramel",
]
