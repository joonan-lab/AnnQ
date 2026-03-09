"""
AnnQ v3.4 - Main Entry Point
Complete rewrite with OOR support
"""

import argparse
import yaml
import sys
import os
import pandas as pd
import numpy as np
from .core import run_classification, run_oor_analysis
from .plotting import plot_results

# -----------------------------------------------------------------------------
# Default Configuration Template v3.4
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = """# AnnQ Configuration File v3.4
# TIP: Use 'annq --init --interactive' for guided setup!

# 1. Annotation Settings (Reference Models)
annotation:
  tool: "celltypist"
  models:
    - "Mouse_Whole_Brain.pkl"
    - "Developing_Mouse_Brain.pkl"
  # hierarchy_file: "/path/to/custom_hierarchy.json"  # Optional

# 2. Input Data
input:
  file_path: "your_data.csv"
  start_col_index: 11          # Column index where probability data starts (1-based)

# 3. Parameters
parameters:
  mode: "standard"             # standard, soft, or adaptive
  threshold: 0.5               # Hard threshold
  soft_threshold: 0.3          # Soft threshold
  auto_suggest: false
  refine_g1: false
  
  g1_refinement_rules:
    lowconf_p1_max: 0.25
    ambiguous_p1_min: 0.3
    ambiguous_ratio_min: 0.6
    ambiguous_delta_max: 0.2
    diffuse_entropy_min: 0.7
  
  generate_threshold_report: false

# 4. Output Settings
output:
  dir: "./output_results"
  save_csv: true

# 5. Visualization Settings
visualization:
  generate_plots: true
  plot_columns:
    - "Age"
    - "Genotype"
    - "Sex"

# 6. OOR Analysis 
oor:
  enabled: true
  cell_id_col: "cell_id"
  cluster_col: "Predicted_Label"
  condition_col: "Genotype"
  baseline_label: "WT"
  comparison_mode: "binary"    # binary, multi, or pairwise
  comparison_groups: []
  min_baseline_cells_per_cluster: 30
  min_comparison_cells_per_cluster: 10
  feature_cols:
    - "P1"
    - "P2"
    - "Delta"
    - "Entropy"
  tail_q: 0.95
  out_col: "OOR_mahal_score"
  save_cell_scores: true
  save_cluster_summary: true
  oor_subdir: "oor"
  generate_plots: true
  plot_settings:
    plot_median_by_cluster: true
    plot_median_vs_tail: true
    plot_distributions: true
    plot_top_n_clusters: 10
    plot_heatmap: true
    plot_effect_size: true
    figure_format: "png"
    figure_dpi: 150
"""

# -----------------------------------------------------------------------------
# Custom Hierarchy Template
# -----------------------------------------------------------------------------
CUSTOM_HIERARCHY_TEMPLATE = """{
  "MyCustomModel": {
    "_metadata": {
      "description": "Custom cell type hierarchy",
      "reference": "Your reference here"
    },
    "CellType1": {
      "Tissue": "Brain",
      "Cell_Class": "Neuron"
    }
  }
}
"""


def create_init_config(filename, force=False, interactive=False):
    """Generates a default configuration file."""
    if not filename.endswith(('.yaml', '.yml')):
        filename += ".yaml"
    
    file_exists = os.path.exists(filename)
    
    if file_exists and not force:
        print(f"Error: File '{filename}' already exists.")
        print(f"Use --force to overwrite: annq --init {filename} --force")
        sys.exit(1)
    
    if interactive:
        config_content = create_interactive_config()
    else:
        config_content = DEFAULT_CONFIG
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    action = "Overwritten" if file_exists else "Created"
    print(f"✓ Success: {action} configuration file '{filename}'.")
    print(f"Run analysis with: annq -c {filename}")


def create_interactive_config():
    """Interactive config generator with guided questions"""
    print("\n" + "="*70)
    print("AnnQ v3.4 Configuration Setup")
    print("="*70)
    
    # Question 1: Classification mode
    print("\nChoose classification mode:")
    print("  1. Standard - Single threshold")
    print("  2. Soft - Capture transitional cells (recommended)")
    print("  3. Adaptive - Auto-optimize thresholds")
    
    while True:
        choice = input("\nEnter choice [1-3]: ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Question 2: Input file
    print("\n" + "-"*70)
    input_file = input(f"Input CSV file path [your_data.csv]: ").strip() or "your_data.csv"
    
    # Question 3: Data start column
    while True:
        start_col = input("Column index where probability data starts [11]: ").strip() or "11"
        try:
            start_col_int = int(start_col)
            if start_col_int > 0:
                break
        except ValueError:
            pass
        print("Please enter a valid positive integer.")
    
    # Question 4: Annotation models
    print("\n" + "-"*70)
    tool = input("Tool name [celltypist]: ").strip() or "celltypist"
    models_input = input("Models (comma-separated) [Mouse_Whole_Brain.pkl]: ").strip()
    models = [m.strip() for m in models_input.split(',')] if models_input else ["Mouse_Whole_Brain.pkl"]
    
    # Question 5: Visualization columns
    print("\n" + "-"*70)
    plot_cols_input = input("Metadata columns (comma-separated) [Age,Genotype,Sex]: ").strip()
    plot_columns = [c.strip() for c in plot_cols_input.split(',')] if plot_cols_input else ["Age", "Genotype", "Sex"]
    
    # Question 6: Output directory
    print("\n" + "-"*70)
    default_output = {'1': "./output_standard", '2': "./output_soft", '3': "./output_adaptive"}
    output_dir = input(f"Output directory [{default_output[choice]}]: ").strip() or default_output[choice]
    
    # Question 7: OOR Analysis
    print("\n" + "-"*70)
    print("OOR (Out-of-Reference) Analysis:")
    print("  Compare experimental conditions (e.g., WT vs mutant)")
    oor_enabled = input("Enable OOR analysis? [y/N]: ").strip().lower() in ['y', 'yes']
    
    oor_config_str = ""
    if oor_enabled:
        condition_col = input("  Condition column [Genotype]: ").strip() or "Genotype"
        baseline_label = input("  Baseline label [WT]: ").strip() or "WT"
        print("\n  Comparison mode:")
        print("    1. Binary - Baseline vs all others")
        print("    2. Multi - Each group separately (RECOMMENDED)")
        print("    3. Pairwise - All combinations")
        comp_choice = input("  Enter choice [1-3, default=2]: ").strip() or "2"
        comp_mode = {'1': 'binary', '2': 'multi', '3': 'pairwise'}.get(comp_choice, 'multi')
        
        oor_config_str = f"""
# 6. OOR Analysis
oor:
  enabled: true
  cell_id_col: "cell_id"
  cluster_col: "Predicted_Label"
  condition_col: "{condition_col}"
  baseline_label: "{baseline_label}"
  comparison_mode: "{comp_mode}"
  comparison_groups: []
  min_baseline_cells_per_cluster: 30
  min_comparison_cells_per_cluster: 10
  feature_cols: ["P1", "P2", "Delta", "Entropy"]
  tail_q: 0.95
  out_col: "OOR_mahal_score"
  save_cell_scores: true
  save_cluster_summary: true
  oor_subdir: "oor"
  generate_plots: true
  plot_settings:
    plot_top_n_clusters: 10
    figure_format: "png"
    figure_dpi: 150
"""
    else:
        oor_config_str = "\n# 6. OOR Analysis (disabled)\n"
    
    # Generate config
    mode_map = {'1': 'standard', '2': 'soft', '3': 'adaptive'}
    mode = mode_map[choice]
    auto_suggest = (choice in ['2', '3'])
    refine_g1 = (choice in ['2', '3'])
    
    config = f"""# AnnQ Configuration File v3.4

# 1. Annotation Settings
annotation:
  tool: "{tool}"
  models:
"""
    for model in models:
        config += f'    - "{model}"\n'
    
    config += f"""
# 2. Input Data
input:
  file_path: "{input_file}"
  start_col_index: {start_col_int}

# 3. Parameters
parameters:
  mode: "{mode}"
  threshold: 0.5
  soft_threshold: 0.3
  auto_suggest: {str(auto_suggest).lower()}
  refine_g1: {str(refine_g1).lower()}
  generate_threshold_report: {str(auto_suggest).lower()}

# 4. Output Settings
output:
  dir: "{output_dir}"
  save_csv: true

# 5. Visualization Settings
visualization:
  generate_plots: true
  plot_columns:
"""
    for col in plot_columns:
        config += f'    - "{col}"\n'
    
    config += oor_config_str
    
    # Summary
    print("\n" + "="*70)
    print("Configuration Summary:")
    print(f"  Mode: {mode}")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_dir}")
    print(f"  OOR: {'Yes' if oor_enabled else 'No'}")
    print("="*70 + "\n")
    
    return config


def create_hierarchy_template(filename, force=False):
    """Generates a custom hierarchy JSON template."""
    if not filename.endswith('.json'):
        filename += ".json"
    
    file_exists = os.path.exists(filename)
    
    if file_exists and not force:
        print(f"Error: File '{filename}' already exists.")
        print(f"Use --force to overwrite")
        sys.exit(1)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(CUSTOM_HIERARCHY_TEMPLATE)
    
    print(f"Success: Created hierarchy template '{filename}'.")


def main():
    parser = argparse.ArgumentParser(
        description="AnnQ v3.4: Annotation QC with OOR Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--init", nargs='?', const="config.yaml", metavar="FILENAME")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--init-hierarchy", nargs='?', const="custom_hierarchy.json")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--oor", action="store_true", help="Run OOR analysis")
    
    args = parser.parse_args()

    # Mode 1: Initialize Config
    if args.init:
        create_init_config(args.init, force=args.force, interactive=args.interactive)
        sys.exit(0)
    
    # Mode 2: Initialize Hierarchy
    if args.init_hierarchy:
        create_hierarchy_template(args.init_hierarchy, force=args.force)
        sys.exit(0)

    # Mode 3: Run Analysis
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    print(f"Running AnnQ v3.4 with config: {args.config}")
    
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)

    # OOR Analysis
    if args.oor:
        run_oor_analysis(config)
        print("\nAnnQ v3.4 OOR analysis completed.")
        sys.exit(0)

    # Classification
    result_df = run_classification(config)
    
    # Plots
    if config['visualization']['generate_plots']:
        prob_mat = None
        if config['parameters'].get('generate_threshold_report', False):
            try:
                input_file = config['input']['file_path']
                data_start_idx = config['input']['start_col_index'] - 1
                df_temp = pd.read_csv(input_file, low_memory=False)
                prob_mat = df_temp.iloc[:, data_start_idx:].values
                if np.isnan(prob_mat).any():
                    prob_mat = np.nan_to_num(prob_mat, nan=0.0)
            except Exception as e:
                print(f"Warning: Could not reload probability matrix: {e}")
        
        plot_results(result_df, config, prob_mat=prob_mat)
    
    print("\nAnnQ v3.4 Pipeline Completed.")


if __name__ == "__main__":
    main()