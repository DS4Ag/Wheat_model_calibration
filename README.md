# Wheat Model Calibration

This repository provides a modular, reproducible Python framework for advanced crop model calibration and benchmarking—designed for analyzing multi-trait wheat data across diverse environments. Each analysis type (e.g., clustering, ecotype calibration) is organized as a standalone module for maximum extensibility and reusability.

## Project Structure

```text
Wheat_model_calibration/
├── data/                          # Input data for each analysis module
│   ├── clustering/
│   ├── DSSAT48/
│   └── ecotype_calibration/
├── output/                        # Output results (figures, summaries) per module
│   ├── clustering/
│   └── ecotype_calibration/
├── scripts/                       # Orchestrated entry-point scripts for each workflow
│   ├── generate_cluster_figure.py
│   ├── generate_ecotype_heatmaps.py
│   └── analyze_feature_contributions.py
├── src/                           # Core reusable code for all analyses
│   ├── clustering/
│   │   ├── config_paths.py
│   │   ├── feature_contributions.py
│   │   ├── integrated_cluster_figure.py
│   │   ├── load_data.py
│   │   ├── manual_offsets.py
│   │   ├── plot_style.py
│   │   └── variable_mapping.py
│   └── ecotype_calibration/
│       ├── config_paths.py
│       ├── heatmap_figure.py
│       ├── plot_style.py
│       └── variable_mapping.py
│   ├── metrics/
│   ├── data_preparation.py
│   ├── utils.py
│   └── __init__.py
├── requirements.txt               # Python dependencies (all modules)
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/wheat_model_calibration.git
cd wheat_model_calibration
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Analysis Modules

Each analysis module in `scripts/` is fully independent—results are saved under `output/`.  
The workflow for each is: **extract → calculate → plot** (modular steps, re-used across modules).

| Analysis              | Source Folder              | Script to Run                          | Output Folder               |
|-----------------------|---------------------------|----------------------------------------|---------------------------|
| Clustering            | `src/clustering/`         | `scripts/generate_cluster_figure.py`   | `output/clustering/`       |
| Ecotype Calibration   | `src/ecotype_calibration/`| `scripts/generate_ecotype_heatmaps.py` | `output/ecotype_calibration/` |

### Clustering Analysis

Performs:
- **Dimensionality reduction** (PCA)
- **Hierarchical & KMeans clustering**
- **Integrated cluster visualization**

```bash
python scripts/generate_cluster_figure.py
```

**Outputs:**  
- Integrated cluster figure (`output/clustering/`)
- Cluster assignments CSV

--- 

### Ecotype Calibration Module

Evaluates:
- **Multi-environment ecotype calibration**
- **Four key metrics:** NRMSE, MPE, R² (1:1), Gain
- **4-panel comparative heatmap**

```bash
python scripts/generate_ecotype_heatmaps.py
```

**Outputs:**
- Multi-metric heatmap SVG (`output/ecotype_calibration/`)

## Input Data Requirements

All datasets should be in tabular (CSV or DSSAT OVERVIEW.OUT) form, with numeric trait columns and minimal preprocessing required.  
Config files (YAML) for each calibration subset specify which variables, methods, and metadata to use.