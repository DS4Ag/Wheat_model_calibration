"""
Input and output paths used in ecotype calibration analysis.
"""

import os

# Base directories
DATA_DIR = './data/ecotype_calibration'
OUTPUT_DIR = './output/ecotype_calibration'

# Input subfolders
calibration_subsets = [
    'cultivar_subset_a',
    'cultivar_subset_b',
    'cultivar_subset_c',
    'cultivar_subset_d'
]

# File mapping
overview_files = {
    os.path.join(DATA_DIR, subset, 'OVERVIEW.OUT'): subset
    for subset in calibration_subsets
}

# Output figure
final_figure_output = os.path.join(OUTPUT_DIR, 'cluster-evaluation_nrmse-mpe-r2-gain_ONE-ROW_WW-23.svg')