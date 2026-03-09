# AnnQ

**Quality control and classification tool for single-cell annotation results, with post-hoc out-of-reference (OOR) analysis for perturbation studies.**

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

AnnQ evaluates the quality of single-cell annotation outputs by classifying cells into interpretable groups based on predicted probability distributions. It also provides a **post-hoc Out-of-Reference (OOR) analysis** mode to quantify how far cells in perturbed or mutant conditions deviate from a reference annotation space.

**Key features:**
- Flexible classification modes: Standard, Soft, and Adaptive
- Cell-level OOR scoring using Mahalanobis distance
- Cluster-level OOR summary with statistical tests
- Multiple comparison modes: binary, multi-group, and pairwise
- Fully configurable via YAML — no hardcoded parameters
- Enhanced visualizations: heatmaps, fold change plots, distribution comparisons

---

## Prerequisites

- **OS:** macOS, Linux, Windows
- **Python:** 3.9 or higher
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

---

## Installation

**Step 1. Clone the repository**

```bash
git clone https://github.com/joonan-lab/AnnQ.git
cd AnnQ
```

**Step 2. Create and activate a conda environment**

```bash
conda create -n annq python=3.10 -y
conda activate annq
```

> **Tip:** Once activated, you should see `(annq)` at the start of your terminal prompt.

**Step 3. Install dependencies and AnnQ**

```bash
conda install -c conda-forge numpy pandas scipy matplotlib seaborn pyyaml -y
pip install -e .
pip install celltypist
```

---

## Initial Setup (required once)

After installation, run the following to fetch the latest CellTypist model information and build the internal classification database:

```bash
python src/AnnQ/create_hierarchy_json.py
```

> **Success indicator:** `Success! JSON database saved directly to...`

---

## Quick Start

```bash
# 1. Generate a config file for your project
annq --init my_project.yaml

# 2. Edit my_project.yaml to match your data (see Configuration below)

# 3. Run annotation quality analysis
annq -c my_project.yaml

# 4. (Optional) Run post-hoc OOR analysis
annq --oor -c my_project.yaml
```

---

## Configuration

AnnQ is controlled entirely through a YAML config file. Below is a full annotated example.

```yaml
# ==========================================
# 1. Annotation Settings
# ==========================================
annotation:
  tool: "celltypist"           # Currently fixed to celltypist

  # List all CellTypist model filenames used in your analysis.
  # If multiple models were used, include all of them.
  models:
    - "Mouse_Whole_Brain.pkl"
    - "Developing_Mouse_Brain.pkl"

# ==========================================
# 2. Input Data
# ==========================================
input:
  file_path: "My_Result.csv"   # Path to your CellTypist output CSV

  # Column index where probability values begin.
  # Check in Excel: find the first probability column (e.g. "B cells", "001 Glut")
  # and count from the left (A=1, B=2, ..., K=11).
  start_col_index: 11

# ==========================================
# 3. Parameters
# ==========================================
parameters:
  threshold: 0.5               # Classification probability threshold (0.5 recommended)

# ==========================================
# 4. Output
# ==========================================
output:
  dir: "./output_results"      # Directory where results will be saved
  save_csv: true               # Save classified results as CSV

# ==========================================
# 5. Visualization
# ==========================================
visualization:
  generate_plots: true         # Generate summary plots

  # Metadata columns to include in plots.
  # Must exactly match the column names (headers) in your CSV file.
  plot_columns:
    - "Age"
    - "Genotype"
    - "Brain Region"

# ==========================================
# 6. OOR Analysis
# ==========================================
oor:
  comparison_mode: "multi"           # Options: binary | multi | pairwise
  baseline_label: "WT"
  comparison_groups: []              # Leave empty for auto-detection

  min_baseline_cells_per_cluster: 30
  min_comparison_cells_per_cluster: 10

  feature_cols:
    - "P1"
    - "P2"
    - "Delta"
    - "Entropy"

  tail_q: 0.95

  generate_plots: true
  plot_settings:
    plot_top_n_clusters: 10
    figure_format: "png"
    figure_dpi: 150
```

---

## Classification Modes

| Mode | Description |
|------|-------------|
| **Standard** | Single threshold (default: 0.5) |
| **Soft** | Two-tier thresholds (e.g. 0.5 / 0.3); captures weak but meaningful secondary identities |
| **Adaptive** | Automatically suggests thresholds based on data distribution (IQR-based) |

---

## Cell Groups

### Core Groups (hard assignments)

| Group | Definition | Interpretation | Recommended Action |
|-------|-----------|----------------|--------------------|
| G0 | Single clear hit | High-confidence annotation | ✅ Use in analysis |
| G1 | No clear hit | Too ambiguous to classify (low quality) | ❌ Filter out |
| G2 | Multi-hit within same `Cell_Class` | Same-lineage ambiguity (e.g. Excitatory neuron type A vs B) | ⚠️ Usable; merge if needed |
| G3 | Multi-hit across `Cell_Class` | Cross-lineage / transitional (e.g. neuron vs glia) | ❓ Suspect doublet |

### Soft Mode Groups

| Group | Interpretation |
|-------|---------------|
| G2s | Weak same-lineage signal |
| G3s | Transitional or mixed identity |

### G1 Refinement (optional)

| Subgroup | Criteria | Interpretation |
|----------|---------|----------------|
| G1a_LowConf | P1 < 0.25 | Likely low information content |
| G1b_Ambiguous | P1 ≥ 0.3, P2/P1 ≥ 0.6 | Biologically transitional |
| G1c_Diffuse | High entropy | Broad, unresolved identity |

---

## Core Annotation Metrics

| Metric | Description |
|--------|-------------|
| **P1** | Highest predicted probability |
| **P2** | Second highest predicted probability |
| **Delta** | P1 − P2 |
| **Admixture** | P2 / P1 |
| **Entropy (A_Ambiguity)** | Spread of probabilities across cell types |

---

## Out-of-Reference (OOR) Analysis

AnnQ provides a post-hoc analysis mode to quantify how far each cell deviates from a reference annotation space — particularly useful for **mutant or perturbation studies**.

### OOR Score

For each cell, AnnQ computes `OOR_mahal_score` (configurable via `out_col`):

> How atypical a cell's annotation probability structure is, relative to baseline cells within the same cluster.

This score is computed as a **Mahalanobis distance** using the following features (configurable): P1, Delta, Admixture, and Entropy. Cells are classified as *out-of-reference* if their OOR score exceeds the **tail quantile (default: 95th percentile)** of baseline cells within the same cluster, ensuring that baseline heterogeneity is respected.

### Comparison Modes

| Mode | Behavior |
|------|----------|
| `binary` | Baseline vs. all others combined |
| `multi` | Baseline vs. each comparison group separately |
| `pairwise` | All group combinations (e.g. WT–Het, WT–Hom, Het–Hom) |

### Output Files

```
outdir/
└── oor/
    ├── oor_results.csv.gz                 # Per-cell OOR scores
    ├── oor_cluster_summary.csv            # Per-cluster statistics
    ├── fig_oor_median_by_cluster.png      # Median comparison
    ├── fig_oor_heatmap.png                # Cluster × Condition heatmap
    ├── fig_oor_fold_change.png            # Effect size plot
    ├── fig_oor_median_vs_tail.png         # Shift vs. enrichment
    └── fig_oor_distributions_top10.png    # Top affected clusters
```

Per-cluster statistics include: `n_baseline`, `n_comparison`, `median_baseline`, `median_comparison`, `delta_median`, `fold_change`, `tail_frac_baseline`, `tail_frac_comparison`, `tail_enrichment`, `p_mannwhitney`, `p_ks`.

### Example Configurations

<details>
<summary>Genotype study (WT vs Het vs Hom)</summary>

```yaml
oor:
  enabled: true
  condition_col: "Genotype"
  baseline_label: "WT"
  comparison_mode: "multi"
  comparison_groups: ["Het", "Hom"]
```
</details>

<details>
<summary>Treatment study (multiple timepoints)</summary>

```yaml
oor:
  enabled: true
  condition_col: "Timepoint"
  baseline_label: "Day0"
  comparison_mode: "multi"
  comparison_groups: ["Day3", "Day7", "Day14"]
```
</details>

<details>
<summary>Pairwise comparisons</summary>

```yaml
oor:
  enabled: true
  condition_col: "Treatment"
  baseline_label: "Control"
  comparison_mode: "pairwise"
  comparison_groups: ["Treatment_A", "Treatment_B"]
```
</details>

---

## Interpreting Results

Open the output file `classified_results.csv.gz` and inspect the `group` column.

| Group | Status | Description | Action |
|-------|--------|-------------|--------|
| G0 | ✅ Clear | Single cell type confidently identified | Use in analysis |
| G1 | ❌ Uncertain | Probability too low to classify | Filter out |
| G2 | ⚠️ Similar | Multiple hits within the same lineage | Usable; merge if needed |
| G3 | ❓ Diff Class | Multiple hits across different lineages | Suspect doublet |

---

## FAQ

**Q. `command not found: annq` error**

Your virtual environment is likely not activated. Check that `(annq)` appears at the start of your terminal prompt. If not, run:

```bash
source annq/bin/activate
```

**Q. What happens if I set `start_col_index` incorrectly?**

Probability calculations will be incorrect, resulting in only G1 (Uncertain) outputs or a runtime error. Open your CSV in Excel and count which column the first probability value (e.g. "B cells", "001 Glut") appears in (A=1, B=2, ...).

**Q. Where do I find the model name for the `models` field?**

Use the exact `.pkl` filename of the CellTypist model you used during annotation. Typos will prevent the classification database from loading correctly.

---

## Citation

If you use AnnQ in your research, please cite:

```
Lee. et al. (2026). AnnQ: reference-based quantification of cellular abnormality at single-cell resolution. [Preprint]
```

---

## Contact

**Joon An** — [joonan30@korea.ac.kr](mailto:joonan30@korea.ac.kr)