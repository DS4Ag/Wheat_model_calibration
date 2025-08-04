"""
Input and output paths used throughout ecotype calibration analysis.
Handles location references for input data, configuration files, and output directories.
"""

import os

# Base directories for easy relocation or sharing
BASE_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/ecotype_calibration'))
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/ecotype_calibration'))

# Codes representing the calibration subsets
calibration_codes = [
    'cultivar_subset_a', 'cultivar_subset_b',
    'cultivar_subset_c', 'cultivar_subset_d'
]

# Mapping subset codes to their OVERSIGHT.OUT and config.yaml locations
input_files = {code: os.path.join(BASE_DATA, code, 'OVERVIEW.OUT') for code in calibration_codes}
yaml_files = {code: os.path.join(BASE_DATA, code, 'config.yaml') for code in calibration_codes}

output_dir = BASE_OUTPUT

# Output figure file naming pattern (step and treatment auto-filled)
def heatmap_figure_file(step, treatment):
    """
    Returns the path to the output heatmap for a given analysis step and treatment.
    """
    return os.path.join(BASE_OUTPUT, f"{step}_nrmse-mpe-r2-gain_ONE-ROW_{treatment}.svg")
