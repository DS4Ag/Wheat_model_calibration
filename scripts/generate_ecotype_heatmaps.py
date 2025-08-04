"""
Run the ecotype calibration heatmap generator from script.
"""

import sys
import os

# Ensure src/ is in the Python path to import your package modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import load_and_process_overview
from variable_mapping import treatments_dic, variable_name_mapping
from metrics import compute_grouped_nrmse, calculate_mpe, calculate_r2_1to1, calculate_gain
from data_preparation import prepare_metrics_data
from ecotype_calibration.heatmap_figure import create_heatmap_figure  # Import plotting function (Step 4)

# ---------------------------
# Define input/output paths and calibration sets to process
# ---------------------------
BASE_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/ecotype_calibration'))
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/ecotype_calibration'))

# List of calibration subfolders corresponding to your different ecotype subsets
calibration_code_list = [
    'cultivar_subset_a', 'cultivar_subset_b',
    'cultivar_subset_c', 'cultivar_subset_d'
]

# Mapping each calibration code to its OVERVIEW.OUT data file
input_files = {code: os.path.join(BASE_DATA, code, 'OVERVIEW.OUT') for code in calibration_code_list}

# Mapping each calibration code to its config.yaml file with calibration metadata
yaml_files = {code: os.path.join(BASE_DATA, code, 'config.yaml') for code in calibration_code_list}

# ---------------------------
# Variables and treatment selection for visualization
# ---------------------------
treatment = 'WW-23'

SELECTED_VARIABLES = [
    'Anthesis (DAP)', 'Maturity (DAP)', 'Grain yield (kg/ha)', 'Grain unit weight (g)', 'Grains m2',
    'Harvest index', 'Maximum leaf area index', 'Aboveground biomass at maturity (kg/ha)',
    'Vegetative biomass at maturity (kg/ha)', 'Grain nitrogen content (kg/ha)', 'Grain nitrogen content (%)'
]

if __name__ == '__main__':
    # ---------------------------
    # STEP 1: Extract and load overview data for all calibration subsets
    # ---------------------------
    overview_data = load_and_process_overview(
        calibration_code_list,
        BASE_DATA,
        treatments_dic,
        variable_name_mapping
    )

    # ---------------------------
    # STEP 2: Calculate all required metrics grouped by key columns
    # ---------------------------
    group_cols = ['variable', 'calibration_method', 'short_label', 'long_label', 'treatment']

    nrmse_data = compute_grouped_nrmse(overview_data, group_cols, norm_method='mean')

    # Extract unique calibration labels for filtering and plotting
    selected_short_labels_list = nrmse_data['short_label'].unique().tolist()
    selected_short_labels = selected_short_labels_list  # Can customize if needed

    mpe_data = calculate_mpe(overview_data, group_cols)
    r2_data = calculate_r2_1to1(overview_data, group_cols)
    gain_data = calculate_gain(overview_data, group_cols)

    # ---------------------------
    # STEP 3: Prepare the metric data for heatmap plotting, applying consistent sorting and pivoting
    # ---------------------------
    pivot_table_ordered_trt, mpe_filtered_ordered, r2_filtered_ordered, gain_filtered_ordered = prepare_metrics_data(
        nrmse_data, mpe_data, r2_data, gain_data,
        treatment, SELECTED_VARIABLES, selected_short_labels
    )

    # ---------------------------
    # STEP 4: Create and save the 4-panel heatmap figure visualizing all metrics
    # ---------------------------
    create_heatmap_figure(
        pivot_table_ordered_trt,
        mpe_filtered_ordered,
        r2_filtered_ordered,
        gain_filtered_ordered,
        output_dir=BASE_OUTPUT,
        calibration_step='cluster-evaluation',
        treatment=treatment
    )

