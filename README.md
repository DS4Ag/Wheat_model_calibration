# Wheat Modeling Calibration

This repository provides a modular and reproducible framework for analyzing multi-trait crop data, with a focus on dimensionality reduction and clustering methods. It is structured for extensibility, allowing new analysis types to be added as separate modules.

---

## Project Structure

```

crop\_modeling\_calibration/
├── data/                      # Input data files organized by analysis type
│   └── clustering/
├── plots/                     # Output figures and data
│   └── clustering/
├── scripts/                   # Execution scripts for each analysis module
│   └── generate\_cluster\_figure.py
├── src/
│   └── clustering/            # Core logic for the clustering workflow
│       ├── plot\_cluster\_figure.py
│       ├── config\_paths.py
│       ├── plot\_style.py
│       └── manual\_offsets.py
├── requirements.txt           # Python dependencies
└── README.md                  # This file

````

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crop_modeling_calibration.git
cd crop_modeling_calibration
````

### 2. Set up environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Run analysis

Each analysis module has its own script in the `scripts/` folder. The main output (e.g., figures and CSVs) will be saved in the corresponding `plots/` folder.

---

## Analysis Modules Overview

| Analysis      | Folder (`src/`)   | Script to Run                        | Output Location     |
| ------------- | ----------------- | ------------------------------------ | ------------------- |
| Clustering    | `src/clustering/` | `scripts/generate_cluster_figure.py` | `plots/clustering/` |
| (more coming) | —                 | —                                    | —                   |

---

## Clustering Analysis

This module performs:

* **PCA** with automatic component selection
* **Hierarchical clustering** with dendrograms
* **KMeans clustering** on PCA-reduced data
* Combined **panel figure** showing explained variance, dendrograms, and cluster scatter plots

### Files

| File                                    | Purpose                                                           |
| --------------------------------------- | ----------------------------------------------------------------- |
| `scripts/generate_cluster_figure.py`    | Main script to run the clustering workflow                        |
| `src/clustering/plot_cluster_figure.py` | Core function (`generate_integrated_cluster_figure`) for plotting |
| `src/clustering/config_paths.py`        | Defines input data paths, output paths, and dataset labels        |
| `src/clustering/plot_style.py`          | Contains plot settings (fonts, colors, sizes)                     |
| `src/clustering/manual_offsets.py`      | Optional: manually adjusts text label positions in PCA plots      |

### How to Run

```bash
python scripts/generate_cluster_figure.py
```

This will:

* Load the input datasets from `data/clustering/`
* Generate the composite figure
* Export:

  * A `.png` file with the final figure to `plots/clustering/`
  * A `.csv` file with cluster assignments to `plots/clustering/`

### Input Data Format

Each dataset in `data/clustering/` should be a CSV file with numeric trait columns and optional metadata (e.g., `genotype`, `entry`, `treatment`, `season`). No preprocessing is needed; missing values are handled internally.

---
