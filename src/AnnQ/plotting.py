import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Dict, List


sns.set_theme(style="whitegrid")
PALETTE = {
    "G0": "#2E7D32",           # 초록 (clear single hit)
    "G1": "#757575",           # 중간 회색 (general ambiguous)
    "G1a_LowConf": "#D32F2F",  # 빨강 (low quality - needs filtering)
    "G1b_Ambiguous": "#1976D2", # 진한 파랑 (transitional - biologically interesting!)
    "G1c_Diffuse": "#FFA726",  # 주황 (diffuse across many types)
    "G2": "#1565C0",           # 진한 파랑 (hard multi-hit, same class)
    "G2s": "#64B5F6",          # 연한 파랑 (soft multi-hit, same class)
    "G3": "#EF6C00",           # 진한 주황 (hard multi-hit, diff class)
    "G3s": "#FFB74D",          # 연한 주황 (soft multi-hit, diff class)
}

# AnnQ 분류 결과 컬럼명
ANNQ_GROUP_COL = "annq_group"

def generate_threshold_report(prob_mat, output_dir):
    """
    다양한 threshold 조합에서의 분류 결과를 미리 계산해서 보고서 생성
    유저가 optimal cutoff를 선택할 수 있도록
    """
    print("  Generating threshold exploration report...")
    
    threshold_grid = {
        'hard': [0.4, 0.45, 0.5, 0.55, 0.6],
        'soft': [0.2, 0.25, 0.3, 0.35, 0.4]
    }
    
    results = []
    n_cells = len(prob_mat)
    
    for hard_t in threshold_grid['hard']:
        for soft_t in threshold_grid['soft']:
            if soft_t >= hard_t:
                continue
            
            # Quick classification
            hard_hits = (prob_mat >= hard_t).sum(axis=1)
            soft_hits = (prob_mat >= soft_t).sum(axis=1)
            
            result = {
                'hard_threshold': hard_t,
                'soft_threshold': soft_t,
                'G0_pct': (hard_hits == 1).sum() / n_cells * 100,
                'G1_pct': ((hard_hits == 0) & (soft_hits < 2)).sum() / n_cells * 100,
                'G2_plus_pct': (hard_hits >= 2).sum() / n_cells * 100,
                'G2s_plus_pct': ((hard_hits == 0) & (soft_hits >= 2)).sum() / n_cells * 100,
            }
            results.append(result)
    
    if not results:
        print("  Warning: No valid threshold combinations found")
        return
    
    # Generate heatmap
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['G0_pct', 'G1_pct', 'G2_plus_pct', 'G2s_plus_pct']
    titles = ['G0 (Single Clear Hit)', 'G1 (Low Confidence)', 
              'G2+ (Hard Multi-hit)', 'G2s+ (Soft Multi-hit)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        pivot = results_df.pivot(
            index='hard_threshold', 
            columns='soft_threshold', 
            values=metric
        )
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, 
                    cbar_kws={'label': 'Percentage (%)'})
        ax.set_title(title)
        ax.set_xlabel('Soft Threshold')
        ax.set_ylabel('Hard Threshold')
    
    plt.suptitle('Threshold Exploration: Cell Distribution Across Settings', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_exploration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save table
    results_df.to_csv(os.path.join(output_dir, 'threshold_grid.csv'), index=False)
    
    print(f"  Threshold exploration report saved to {output_dir}")
    print("  Review 'threshold_exploration.png' to choose optimal thresholds")


def plot_metadata_bars(df, output_dir, config):
    """Metadata Stacked Bar Plots"""
    plot_cols = config['visualization'].get('plot_columns', [])
    age_order = [
        "E10.5", "E11.5", "E12", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5", "E17.5", "E18.5",
        "P0", "P1", "P4", "P7", "P8", "P9", "P10", "P14", "P28", "P31", "P56", "P56-63", "P84", "P98", "P100", "Unknown"
    ]
    
    # Determine which column to use for plotting
    group_col = 'annq_group_refined' if 'annq_group_refined' in df.columns else ANNQ_GROUP_COL
    
    # Get unique groups and create color palette
    unique_groups = df[group_col].unique()
    plot_palette = {g: PALETTE.get(g, '#999999') for g in unique_groups}
    
    # Determine plot order (G0, G1 variants, G2 variants, etc.)
    group_order = []
    for base in ['G0', 'G1', 'G1a_LowConf', 'G1b_Ambiguous', 'G1c_Diffuse', 
                 'G2', 'G2s', 'G3', 'G3s', 'G4', 'G4s']:
        if base in unique_groups:
            group_order.append(base)
    # Add any remaining groups not in the standard list
    for g in unique_groups:
        if g not in group_order:
            group_order.append(g)
    
    for col in plot_cols:
        if col not in df.columns: continue
        print(f"  Generating bar plot for {col}...")
        
        plt.figure(figsize=(10, 6))
        df_plot = df.copy()
        df_plot[col] = df_plot[col].fillna("Unknown").astype(str)
        
        unique_vals = df_plot[col].unique()
        if col == "Age":
            order = [x for x in age_order if x in unique_vals] + [x for x in unique_vals if x not in age_order]
        else:
            order = sorted(unique_vals)

        # 100% Stacked Bar
        df_subset = df_plot[df_plot[col].isin(order)]
        
        # Reverse group order for stacking (so G4 is on top)
        hue_order_reversed = [g for g in reversed(group_order) if g in df_subset[group_col].unique()]
        
        sns.histplot(
            data=df_subset, x=col, hue=group_col, multiple='fill', 
            palette=plot_palette, hue_order=hue_order_reversed,
            shrink=0.8, edgecolor='none'
        )
        
        plt.gca().set_xticks(range(len(order)))
        plt.gca().set_xticklabels(order, rotation=45, ha='right')
        plt.title(f"Classification Distribution by {col}")
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"bar_{col}.png"), dpi=300)
        plt.close()

def plot_qc_distributions(df, output_dir):
    """Boxplots and Density plots for QC metrics"""
    metrics = ['AQ_Delta', 'AQ_TotalSum', 'A_Assurance', 'A_Ambiguity', 'A_Admixture']
    
    # Determine which column to use
    group_col = 'annq_group_refined' if 'annq_group_refined' in df.columns else ANNQ_GROUP_COL
    unique_groups = sorted(df[group_col].unique())
    plot_palette = {g: PALETTE.get(g, '#999999') for g in unique_groups}
    
    for metric in metrics:
        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df, x=group_col, y=metric, 
            hue=group_col, palette=plot_palette, legend=False,
            order=unique_groups
        )
        plt.title(f"{metric} Distribution by Group")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_{metric}.png"), dpi=300)
        plt.close()
        
        # Density Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=df, x=metric, hue=group_col, 
            palette=plot_palette, fill=True, alpha=0.3, 
            common_norm=False, warn_singular=False
        )
        plt.title(f"{metric} Density by Group")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"density_{metric}.png"), dpi=300)
        plt.close()

def plot_scatter_analysis(df, output_dir):
    """Scatter plots for correlation analysis"""
    print("  Generating scatter plots...")
    
    group_col = 'annq_group_refined' if 'annq_group_refined' in df.columns else ANNQ_GROUP_COL
    unique_groups = df[group_col].unique()
    plot_palette = {g: PALETTE.get(g, '#999999') for g in unique_groups}
    
    # 1. Delta vs Assurance
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='A_Assurance', y='AQ_Delta', hue=group_col, 
                    palette=plot_palette, alpha=0.5, s=15)
    plt.axhline(0.5, ls='--', color='gray', alpha=0.5)
    plt.axvline(0.5, ls='--', color='gray', alpha=0.5)
    plt.title("Delta vs Assurance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_delta_vs_assurance.png"), dpi=300)
    plt.close()

    # 2. Delta vs Admixture
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='A_Admixture', y='AQ_Delta', hue=group_col, 
                    palette=plot_palette, alpha=0.5, s=15)
    plt.axvline(0.5, ls='--', color='red', alpha=0.5)
    y_pos = df['AQ_Delta'].max() * 0.9 if not df['AQ_Delta'].empty else 0.9
    plt.text(0.55, y_pos, "Doublet Zone", color='red')
    plt.title("Delta vs Admixture")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_delta_vs_admixture.png"), dpi=300)
    plt.close()

    # 3. TotalSum vs Ambiguity
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='A_Ambiguity', y='AQ_TotalSum', hue=group_col, 
                    palette=plot_palette, alpha=0.5, s=15)
    plt.title("Total Sum vs Ambiguity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_totalsum_vs_ambiguity.png"), dpi=300)
    plt.close()

def plot_g0_analysis(df, output_dir):
    """Deep dive into G0 cells"""
    g0_df = df[df[ANNQ_GROUP_COL] == 'G0']
    if len(g0_df) == 0: return
    
    print("  Generating G0 analysis plots...")
    
    # 1. G0 Delta Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=g0_df, x='AQ_Delta', color=PALETTE['G0'], binwidth=0.02)
    plt.axvline(0.3, color='orange', linestyle='--')
    plt.axvline(0.5, color='red', linestyle='--')
    
    ylim = plt.ylim()[1]
    plt.text(0.32, ylim*0.9, "Moderate", color='orange')
    plt.text(0.52, ylim*0.9, "High Conf", color='red')
    plt.title(f"G0 Cells: Delta Score Distribution (n={len(g0_df)})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "G0_delta_histogram.png"), dpi=300)
    plt.close()

    # 2. G0 Quality Landscape (Hexbin)
    plt.figure(figsize=(10, 8))
    try:
        plt.hexbin(g0_df['A_Assurance'], g0_df['AQ_Delta'], gridsize=50, cmap='plasma', mincnt=1)
        cb = plt.colorbar(label='Count')
        plt.axhline(0.5, color='white', linestyle='--')
        plt.axvline(0.5, color='white', linestyle='--')
        plt.title("G0 Cells: Quality Landscape")
        plt.xlabel("Assurance (P1)")
        plt.ylabel("Delta (P1 - P2)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "G0_quality_landscape.png"), dpi=300)
    except Exception as e:
        print(f"  Warning: Skipping hexbin plot due to data issues: {e}")
    plt.close()

def plot_summary_table_image(df, output_dir):
    """Generate QC Summary Table as an image"""
    group_col = 'annq_group_refined' if 'annq_group_refined' in df.columns else ANNQ_GROUP_COL
    
    summary = df.groupby(group_col)[['AQ_Delta', 'A_Assurance', 'A_Ambiguity']].mean().round(3)
    summary['Count'] = df[group_col].value_counts()
    summary = summary.reset_index()
    
    fig, ax = plt.subplots(figsize=(12, len(summary) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    cols = ['Group', 'Count', 'Mean Delta', 'Mean Assurance', 'Mean Ambiguity']
    
    # Sort by group name
    summary = summary.sort_values(group_col)
    
    for _, row in summary.iterrows():
        table_data.append([
            row[group_col], 
            int(row['Count']), 
            row['AQ_Delta'], 
            row['A_Assurance'], 
            row['A_Ambiguity']
        ])
        
    if not table_data: return

    table = ax.table(cellText=table_data, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("QC Metrics Summary Table", pad=20, fontweight='bold')
    plt.savefig(os.path.join(output_dir, "qc_summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_p1_p2_distributions(df, output_dir):
    """
    P1 and P2 distribution plots
    
    Generates:
    1. Density plots for P1 and P2 distributions
    2. Scatter plot of P1 vs P2
    3. Colored by classification groups
    """
    if 'P1' not in df.columns or 'P2' not in df.columns:
        print("  Warning: P1/P2 columns not found, skipping P1/P2 plots")
        return
    
    group_col = 'annq_group_refined' if 'annq_group_refined' in df.columns else ANNQ_GROUP_COL
    unique_groups = df[group_col].unique()
    plot_palette = {g: PALETTE.get(g, '#999999') for g in unique_groups}
    
    # 1. Density plots for P1 and P2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # P1 density
    for group in sorted(unique_groups):
        subset = df[df[group_col] == group]
        if len(subset) > 0:
            axes[0].hist(subset['P1'], bins=50, alpha=0.5, 
                        label=f"{group} (n={len(subset)})",
                        color=plot_palette[group], density=True)
    
    axes[0].set_xlabel('P1 (Highest Probability)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('P1 Distribution by Group', fontsize=13, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # P2 density
    for group in sorted(unique_groups):
        subset = df[df[group_col] == group]
        if len(subset) > 0:
            axes[1].hist(subset['P2'], bins=50, alpha=0.5,
                        label=f"{group} (n={len(subset)})",
                        color=plot_palette[group], density=True)
    
    axes[1].set_xlabel('P2 (Second Highest Probability)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('P2 Distribution by Group', fontsize=13, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p1_p2_density.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot: P1 vs P2
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 2a. All groups
    for group in sorted(unique_groups):
        subset = df[df[group_col] == group]
        if len(subset) > 0:
            axes[0].scatter(subset['P2'], subset['P1'], 
                          alpha=0.4, s=10,
                          label=f"{group} (n={len(subset)})",
                          color=plot_palette[group])
    
    # Add reference lines
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='P1=P2')
    axes[0].axhline(0.5, color='red', linestyle=':', alpha=0.3, label='P1=0.5')
    axes[0].axvline(0.3, color='blue', linestyle=':', alpha=0.3, label='P2=0.3')
    
    axes[0].set_xlabel('P2 (Second Highest)', fontsize=12)
    axes[0].set_ylabel('P1 (Highest)', fontsize=12)
    axes[0].set_title('P1 vs P2: All Groups', fontsize=13, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    
    # 2b. Hexbin density (overview)
    hb = axes[1].hexbin(df['P2'], df['P1'], gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='P1=P2')
    axes[1].axhline(0.5, color='red', linestyle=':', alpha=0.5, label='P1=0.5')
    axes[1].axvline(0.3, color='blue', linestyle=':', alpha=0.5, label='P2=0.3')
    
    axes[1].set_xlabel('P2 (Second Highest)', fontsize=12)
    axes[1].set_ylabel('P1 (Highest)', fontsize=12)
    axes[1].set_title('P1 vs P2: Density', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.colorbar(hb, ax=axes[1], label='Cell Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p1_vs_p2_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  P1/P2 distribution plots generated")

def plot_results(df, config, prob_mat=None):
    """Main plotting function"""
    if not config['visualization']['generate_plots']:
        return

    output_dir = os.path.join(config['output']['dir'], "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # 경고 무시 설정 (cleaner output)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Generate threshold exploration if requested and prob_mat available
    if config['parameters'].get('generate_threshold_report', False) and prob_mat is not None:
        generate_threshold_report(prob_mat, output_dir)
    
    plot_metadata_bars(df, output_dir, config)
    plot_qc_distributions(df, output_dir)
    plot_scatter_analysis(df, output_dir)
    plot_p1_p2_distributions(df, output_dir)  # NEW: P1/P2 plots
    plot_g0_analysis(df, output_dir)
    plot_summary_table_image(df, output_dir)
    
    print(f"All plots saved to {output_dir}")

##########################
## OOR

def plot_oor_results(
    scored_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
    *,
    genotype_col: str,
    cluster_col: str,
    wt_label: str,
    score_col: str = "OOR_mahal_clusterWT",
    tail_q: float = 0.90,
    max_clusters_to_plot: int = 30,
    plots_cfg: Optional[Dict] = None,
):
    """Generate standard OOR plots into output_dir.

    This function intentionally produces multiple views because Mahalanobis-based
    scores typically have heavy tails (a few extreme cells can squash the scale).

    Creates (depending on plots_cfg):
      - fig_oor_violin_by_cluster.png (raw)
      - fig_oor_violin_by_cluster_log.png (log y)
      - fig_oor_violin_by_cluster_clip.png (y clipped at a quantile)
      - fig_oor_median_by_cluster.png (cluster-wise medians by genotype)
      - fig_oor_median_vs_tail.png (cluster summary: delta_median vs tail fraction)
      - fig_oor_tail_enrichment.png (tail enrichment barplot)
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_cfg = plots_cfg or {}

    # ---- plot preset handling ----
    # preset: 'minimal' (default) | 'standard' | 'all'
    preset = str(plots_cfg.get("preset", "minimal")).lower().strip()
    if preset not in {"minimal", "standard", "all"}:
        preset = "minimal"

    if preset == "minimal":
        plots_cfg["make_violin"] = bool(plots_cfg.get("make_violin", False))
        plots_cfg["make_violin_log"] = bool(plots_cfg.get("make_violin_log", False))
        plots_cfg["make_violin_clip"] = bool(plots_cfg.get("make_violin_clip", False))
        plots_cfg["make_tail_bar"] = bool(plots_cfg.get("make_tail_bar", False))
        plots_cfg["make_median_plot"] = bool(plots_cfg.get("make_median_plot", True))
        plots_cfg["make_median_vs_tail"] = bool(plots_cfg.get("make_median_vs_tail", True))
    elif preset == "standard":
        plots_cfg["make_violin"] = bool(plots_cfg.get("make_violin", False))
        plots_cfg["make_violin_log"] = bool(plots_cfg.get("make_violin_log", False))
        plots_cfg["make_violin_clip"] = bool(plots_cfg.get("make_violin_clip", False))
        plots_cfg["make_tail_bar"] = bool(plots_cfg.get("make_tail_bar", False))
        plots_cfg["make_median_plot"] = bool(plots_cfg.get("make_median_plot", True))
        plots_cfg["make_median_vs_tail"] = bool(plots_cfg.get("make_median_vs_tail", True))
    else:  # all
        plots_cfg["make_violin"] = bool(plots_cfg.get("make_violin", False))
        plots_cfg["make_violin_log"] = bool(plots_cfg.get("make_violin_log", False))
        plots_cfg["make_violin_clip"] = bool(plots_cfg.get("make_violin_clip", False))
        plots_cfg["make_tail_bar"] = bool(plots_cfg.get("make_tail_bar", False))
        plots_cfg["make_median_plot"] = bool(plots_cfg.get("make_median_plot", True))
        plots_cfg["make_median_vs_tail"] = bool(plots_cfg.get("make_median_vs_tail", True))


    # ---- cluster subset to keep plots readable ----
    cluster_counts = scored_df[cluster_col].astype(str).value_counts()
    clusters = list(cluster_counts.head(max_clusters_to_plot).index)

    plot_df = scored_df[scored_df[cluster_col].astype(str).isin(clusters)].copy()
    plot_df[cluster_col] = plot_df[cluster_col].astype(str)

    # Stable ordering
    order = clusters

    def _save(fig_path: str):
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    # ---- Helper: violin/box ----
    def _violin_or_box(df: pd.DataFrame, *, title: str, fig_path: str, yscale: Optional[str] = None, ylim: Optional[tuple] = None):
        plt.figure(figsize=(max(10, len(order) * 0.35), 6))
        try:
            sns.violinplot(
                data=df,
                x=cluster_col,
                y=score_col,
                hue=genotype_col,
                order=order,
                cut=0,
                inner="quartile",
                linewidth=0.5,
            )
        except Exception:
            sns.boxplot(
                data=df,
                x=cluster_col,
                y=score_col,
                hue=genotype_col,
                order=order,
            )

        if yscale:
            plt.yscale(yscale)
        if ylim is not None:
            plt.ylim(*ylim)

        plt.title(title)
        plt.xlabel("Cluster")
        plt.ylabel(score_col)
        plt.xticks(rotation=90)
        _save(fig_path)

    # 1) Raw violin
    if plots_cfg.get("make_violin", False):
        _violin_or_box(
            plot_df,
            title=f"Out-of-reference score by cluster (WT='{wt_label}')",
            fig_path=os.path.join(output_dir, "fig_oor_violin_by_cluster.png"),
        )

    # 2) Log-scale version (recommended for heavy-tailed distances)
    if plots_cfg.get("make_violin_log", False):
        # Add a tiny floor so log-scale doesn't break on zeros
        df_log = plot_df.copy()
        df_log[score_col] = np.maximum(df_log[score_col].astype(float), 1e-6)
        _violin_or_box(
            df_log,
            title=f"Out-of-reference score by cluster (log scale; WT='{wt_label}')",
            fig_path=os.path.join(output_dir, "fig_oor_violin_by_cluster_log.png"),
            yscale="log",
        )

    # 3) Clipped y-axis version (shows the 'typical' distribution)
    if plots_cfg.get("make_violin_clip", False):
        q = float(plots_cfg.get("clip_quantile", 0.99))
        # Compute on the plotted subset for a consistent view
        ymax = float(np.nanquantile(plot_df[score_col].astype(float).values, q))
        # Guard against degenerate cases
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = None
        if ymax is not None:
            _violin_or_box(
                plot_df,
                title=f"Out-of-reference score by cluster (clipped at q={q:.2f}; WT='{wt_label}')",
                fig_path=os.path.join(output_dir, "fig_oor_violin_by_cluster_clip.png"),
                ylim=(0, ymax),
            )

    # 4) Cluster-wise median comparison plot (much easier to interpret)
    if plots_cfg.get("make_median_plot", True):
        fig_path = os.path.join(output_dir, "fig_oor_median_by_cluster.png")
        m = (
            plot_df.groupby([cluster_col, genotype_col])[score_col]
            .median()
            .reset_index()
        )
        m[cluster_col] = m[cluster_col].astype(str)

        plt.figure(figsize=(max(10, len(order) * 0.35), 4.8))
        # pointplot draws medians cleanly; we already computed medians to avoid seaborn estimator confusion
        sns.pointplot(
            data=m,
            x=cluster_col,
            y=score_col,
            hue=genotype_col,
            order=order,
            dodge=0.35,
            join=False,
            errorbar=None,
        )
        plt.title(f"Median out-of-reference score by cluster (WT='{wt_label}')")
        plt.xlabel("Cluster")
        plt.ylabel(f"median({score_col})")
        plt.xticks(rotation=90)
        _save(fig_path)

    # 5) Cluster summary scatter: delta_median vs tail fraction
    if plots_cfg.get("make_median_vs_tail", True) and (summary_df is not None) and (not summary_df.empty):
        fig_path = os.path.join(output_dir, "fig_oor_median_vs_tail.png")
        sdf = summary_df.copy()
        if "Cluster" in sdf.columns:
            sdf["Cluster"] = sdf["Cluster"].astype(str)
            sdf = sdf[sdf["Cluster"].isin(order)]

        # Some summaries might not have delta_median if schema changed
        if ("delta_median" in sdf.columns) and ("tail_frac_NonWT" in sdf.columns):
            plt.figure(figsize=(6.5, 5.5))
            sns.scatterplot(data=sdf, x="delta_median", y="tail_frac_NonWT")
            # Annotate each point with cluster id
            for _, row in sdf.iterrows():
                try:
                    plt.text(float(row["delta_median"]), float(row["tail_frac_NonWT"]), str(row["Cluster"]), fontsize=8)
                except Exception:
                    pass
            plt.axhline(y=(1.0 - tail_q), linestyle='--', color='k', linewidth=1, alpha=0.6)
            plt.axvline(x=0.0, linestyle='--', color='k', linewidth=1, alpha=0.6)
            plt.title("Cluster summary: median shift vs tail enrichment")
            plt.xlabel("delta_median (NonWT - WT)")
            plt.ylabel(f"NonWT tail fraction (WT q={tail_q:.2f})")
            _save(fig_path)

    # 6) Tail enrichment barplot (NonWT fraction above WT tail threshold)
    if plots_cfg.get("make_tail_bar", False) and (summary_df is not None) and (not summary_df.empty):
        fig_path = os.path.join(output_dir, "fig_oor_tail_enrichment.png")
        sdf = summary_df.copy()
        if "Cluster" in sdf.columns:
            sdf["Cluster"] = sdf["Cluster"].astype(str)
            sdf = sdf[sdf["Cluster"].isin(order)]
            sdf = sdf.sort_values("tail_frac_NonWT", ascending=False)

        plt.figure(figsize=(max(10, len(sdf) * 0.35), 5))
        sns.barplot(data=sdf, x="Cluster", y="tail_frac_NonWT", order=list(sdf["Cluster"]))
        plt.axhline(y=(1.0 - tail_q), linestyle='--', color='k', linewidth=1, alpha=0.7)
        plt.title(f"Tail enrichment: fraction NonWT above WT q={tail_q:.2f} threshold")
        plt.xlabel("Cluster")
        plt.ylabel("NonWT tail fraction")
        plt.xticks(rotation=90)
        _save(fig_path)

def plot_oor_binary(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    cluster_col: str = "Predicted_Label",
    condition_col: str = "Genotype",
    oor_col: str = "OOR_mahal_score",
    baseline_label: str = "WT",
    output_dir: str = "./oor",
    top_n_clusters: int = 10,
    figure_format: str = "png",
    figure_dpi: int = 150
):
    """Binary mode: Baseline vs all others (combined)."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[OOR Plots] Binary mode: {baseline_label} vs Others")
    
    # Color palette (2 colors only)
    colors = ['#2E7D32', '#D32F2F']  # Green (WT), Red (NonWT)
    color_map = {baseline_label: colors[0], 'Non-Baseline': colors[1]}
    
    # ========================================================================
    # Plot 1: Median OOR by cluster (simple bar plot)
    # ========================================================================
    if not summary_df.empty and 'median_baseline' in summary_df.columns:
        clusters = summary_df['Cluster'].unique()
        x_pos = np.arange(len(clusters))
        
        baseline_medians = summary_df.groupby('Cluster')['median_baseline'].first()
        comparison_medians = summary_df.groupby('Cluster')['median_comparison'].first()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        width = 0.35
        ax.bar(x_pos - width/2, 
               [baseline_medians.get(cl, 0) for cl in clusters],
               width, label=baseline_label, color=colors[0], alpha=0.8)
        ax.bar(x_pos + width/2, 
               [comparison_medians.get(cl, 0) for cl in clusters],
               width, label='Non-Baseline', color=colors[1], alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(clusters, rotation=45, ha='right')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel(f'Median {oor_col}', fontsize=12)
        ax.set_title(f'OOR Score: {baseline_label} vs Others', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_binary_medians.{figure_format}", dpi=figure_dpi)
        plt.close()
        print(f"  ✓ Saved: fig_oor_binary_medians.{figure_format}")
    
    # ========================================================================
    # Plot 2: Simple heatmap (Cluster x Group)
    # ========================================================================
    if not summary_df.empty:
        # Create simple 2-column matrix
        heatmap_data = pd.DataFrame({
            baseline_label: summary_df.groupby('Cluster')['median_baseline'].first(),
            'Non-Baseline': summary_df.groupby('Cluster')['median_comparison'].first()
        })
        
        fig, ax = plt.subplots(figsize=(6, max(6, len(heatmap_data) * 0.3 + 2)))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': f'Median {oor_col}'}, linewidths=0.5)
        ax.set_title('Binary OOR Heatmap', fontsize=14)
        ax.set_ylabel('Cluster', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_binary_heatmap.{figure_format}", dpi=figure_dpi)
        plt.close()
        print(f"  ✓ Saved: fig_oor_binary_heatmap.{figure_format}")
    
    # ========================================================================
    # Plot 3: Fold change bar plot
    # ========================================================================
    if 'fold_change' in summary_df.columns:
        top_clusters = summary_df.nlargest(top_n_clusters, 'fold_change')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(top_clusters))
        ax.barh(y_pos, top_clusters['fold_change'], 
               color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_clusters['Cluster'])
        ax.axvline(1, color='red', linestyle='--', linewidth=1, label='No change')
        ax.set_xlabel('Fold Change (Non-Baseline / Baseline)', fontsize=12)
        ax.set_title(f'Top {top_n_clusters} Clusters by Fold Change', fontsize=14)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_binary_foldchange.{figure_format}", dpi=figure_dpi)
        plt.close()
        print(f"  ✓ Saved: fig_oor_binary_foldchange.{figure_format}")
    
    # ========================================================================
    # Plot 4: Distribution violin plot for top clusters
    # ========================================================================
    if top_n_clusters > 0 and not summary_df.empty:
        top_cluster_list = summary_df.nlargest(top_n_clusters, 'delta_median')['Cluster'].unique()
        
        n_cols = min(3, len(top_cluster_list))
        n_rows = int(np.ceil(len(top_cluster_list) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Create binary label (Baseline vs Non-Baseline)
        df_plot = df.copy()
        df_plot['Binary_Group'] = df_plot[condition_col].apply(
            lambda x: baseline_label if x == baseline_label else 'Non-Baseline'
        )
        
        for i, cluster in enumerate(top_cluster_list[:len(axes)]):
            ax = axes[i]
            sub = df_plot[df_plot[cluster_col] == cluster]
            
            sns.violinplot(data=sub, x='Binary_Group', y=oor_col, 
                          order=[baseline_label, 'Non-Baseline'],
                          palette=color_map, ax=ax, inner='quartile')
            
            # Add median lines
            medians = sub.groupby('Binary_Group')[oor_col].median()
            for j, group in enumerate([baseline_label, 'Non-Baseline']):
                if group in medians.index:
                    ax.hlines(medians[group], j - 0.4, j + 0.4, 
                            colors='red', linewidth=2, linestyles='dashed')
            
            ax.set_title(f'{cluster}', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(oor_col if i % n_cols == 0 else '')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Binary OOR Distribution: Top {top_n_clusters} Clusters', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_binary_distributions.{figure_format}", 
                   dpi=figure_dpi)
        plt.close()
        print(f"  ✓ Saved: fig_oor_binary_distributions.{figure_format}")
    
    print(f"[OOR Plots] Binary plots saved to {output_dir}/")

def plot_oor_multi_group(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    cluster_col: str = "Predicted_Label",
    condition_col: str = "Genotype",
    oor_col: str = "OOR_mahal_score",
    baseline_label: str = "WT",
    comparison_groups: List[str] = None,
    output_dir: str = "./oor",
    top_n_clusters: int = 10,
    figure_format: str = "png",
    figure_dpi: int = 150
):

    """
    Generate comprehensive OOR visualization suite.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level OOR results
    summary_df : pd.DataFrame
        Cluster-level OOR summary
    cluster_col : str
        Cluster/cell type column
    condition_col : str
        Experimental condition column
    oor_col : str
        OOR score column name
    baseline_label : str
        Baseline group label
    comparison_groups : List[str]
        List of comparison group labels
    output_dir : str
        Output directory for plots
    top_n_clusters : int
        Number of top clusters to plot in detail
    figure_format : str
        Output format (png, pdf, svg)
    figure_dpi : int
        Figure resolution
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if comparison_groups is None:
        comparison_groups = df[condition_col].unique().tolist()
        comparison_groups = [g for g in comparison_groups if g != baseline_label]
    
    all_groups = [baseline_label] + comparison_groups
    
    # Color palette
    n_groups = len(all_groups)
    palette = sns.color_palette("Set2", n_groups)
    color_map = dict(zip(all_groups, palette))
    
    print(f"[OOR Plots] Generating visualizations...")
    print(f"[OOR Plots] Groups: {all_groups}")
    
    # ========================================================================
    # Plot 1: Cluster-wise median OOR score
    # ========================================================================
    if "Comparison" in summary_df.columns:
        # Multi-group or pairwise mode
        fig, ax = plt.subplots(figsize=(16, 6))
        
        clusters = summary_df['Cluster'].unique()
        x_pos = np.arange(len(clusters))
        
        for group in all_groups:
            if group == baseline_label:
                # Baseline median (reference)
                group_data = summary_df[summary_df['Baseline'] == baseline_label].groupby('Cluster')['median_baseline'].first()
            else:
                # Comparison medians
                group_data = summary_df[summary_df['Comparison'] == group].set_index('Cluster')['median_comparison']
            
            medians = [group_data.get(cl, np.nan) for cl in clusters]
            ax.scatter(x_pos, medians, label=group, color=color_map[group], s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(clusters, rotation=45, ha='right')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel(f'Median {oor_col}', fontsize=12)
        ax.set_title(f"Median Out-of-Reference Score by Cluster (Baseline='{baseline_label}')", fontsize=14)
        ax.legend(title=condition_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_median_by_cluster.{figure_format}", dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig_oor_median_by_cluster.{figure_format}")
    
    # ========================================================================
    # Plot 2: Heatmap of OOR scores (Cluster x Condition)
    # ========================================================================
    pivot_data = summary_df.pivot_table(
        index='Cluster',
        columns='Comparison',
        values='median_comparison',
        aggfunc='first'
    )
    
    # Add baseline column
    if 'Baseline' in summary_df.columns:
        baseline_medians = summary_df.groupby('Cluster')['median_baseline'].first()
        pivot_data.insert(0, baseline_label, baseline_medians)
    
    fig, ax = plt.subplots(figsize=(10, len(pivot_data) * 0.3 + 2))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': f'Median {oor_col}'}, linewidths=0.5)
    ax.set_title('OOR Score Heatmap (Cluster x Condition)', fontsize=14)
    ax.set_xlabel(condition_col, fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / f"fig_oor_heatmap.{figure_format}", dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig_oor_heatmap.{figure_format}")

    # ========================================================================
    # Plot 3: Effect size (Fold change)
    # ========================================================================
    if 'fold_change' in summary_df.columns and len(comparison_groups) > 0:
        try:  
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get top clusters by delta_median
            top_clusters = summary_df.nlargest(top_n_clusters, 'delta_median')['Cluster'].unique()[:top_n_clusters]
            plot_data = summary_df[summary_df['Cluster'].isin(top_clusters)]
            
            # Prepare data for plotting
            cluster_list = []
            fc_list = []
            group_list = []
            
            for group in comparison_groups:
                group_data = plot_data[plot_data['Comparison'] == group]
                for _, row in group_data.iterrows():
                    cluster_list.append(row['Cluster'])
                    fc_list.append(row['fold_change'])
                    group_list.append(group)
            
            if cluster_list:  # Only plot if we have data
                plot_df = pd.DataFrame({
                    'Cluster': cluster_list,
                    'fold_change': fc_list,
                    'Comparison': group_list
                })
                
                # Create grouped bar plot
                clusters_unique = list(top_clusters)
                x = np.arange(len(clusters_unique))
                width = 0.8 / len(comparison_groups)
                
                for i, group in enumerate(comparison_groups):
                    group_data = plot_df[plot_df['Comparison'] == group]
                    values = [group_data[group_data['Cluster'] == c]['fold_change'].values[0] 
                            if c in group_data['Cluster'].values else 0 
                            for c in clusters_unique]
                    
                    offset = width * i - width * (len(comparison_groups) - 1) / 2
                    ax.bar(x + offset, values, width, label=group, 
                        alpha=0.7, edgecolor='black', linewidth=0.5,
                        color=color_map.get(group, '#999999'))
                
                ax.axhline(1, color='red', linestyle='--', linewidth=1, label='No change (FC=1)')
                ax.set_xlabel('Cluster', fontsize=12)
                ax.set_ylabel('Fold Change (Comparison / Baseline)', fontsize=12)
                ax.set_title(f'OOR Fold Change: Top {top_n_clusters} Clusters (Baseline={baseline_label})', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(clusters_unique, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / f"fig_oor_fold_change.{figure_format}", dpi=figure_dpi, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: fig_oor_fold_change.{figure_format}")
        except Exception as e:
            print(f"  ⚠ Warning: Fold change plot failed: {e}")

    # ========================================================================
    # Plot 4: Median vs Tail Enrichment
    # ========================================================================
    if 'tail_enrichment' in summary_df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for group in comparison_groups:
            group_data = summary_df[summary_df['Comparison'] == group]
            ax.scatter(group_data['delta_median'], group_data['tail_enrichment'], 
                      label=group, color=color_map[group], s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Annotate top clusters
            top_affected = group_data.nlargest(5, 'delta_median')
            for _, row in top_affected.iterrows():
                ax.annotate(row['Cluster'], 
                           (row['delta_median'], row['tail_enrichment']),
                           fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        ax.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No enrichment')
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Δ Median OOR (Comparison - Baseline)', fontsize=12)
        ax.set_ylabel('Tail Enrichment (Comparison / Baseline)', fontsize=12)
        ax.set_title('OOR Shift vs Tail Enrichment', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_median_vs_tail.{figure_format}", dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig_oor_median_vs_tail.{figure_format}")
    
    # ========================================================================
    # Plot 5: Distribution plots for top N clusters
    # ========================================================================
    if top_n_clusters > 0:
        # Identify top affected clusters
        top_clusters = summary_df.nlargest(top_n_clusters, 'delta_median')['Cluster'].unique()
        
        n_cols = min(3, len(top_clusters))
        n_rows = int(np.ceil(len(top_clusters) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, cluster in enumerate(top_clusters[:len(axes)]):
            ax = axes[i]
            sub = df[df[cluster_col] == cluster]
            
            # Violin plot
            sns.violinplot(data=sub, x=condition_col, y=oor_col, order=all_groups,
                          palette=color_map, ax=ax, inner='quartile')
            
            # Add median lines
            medians = sub.groupby(condition_col)[oor_col].median()
            for j, group in enumerate(all_groups):
                if group in medians.index:
                    ax.hlines(medians[group], j - 0.4, j + 0.4, colors='red', linewidth=2, linestyles='dashed')
            
            ax.set_title(f'{cluster}', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(oor_col if i % n_cols == 0 else '')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'OOR Distribution: Top {top_n_clusters} Affected Clusters', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / f"fig_oor_distributions_top{top_n_clusters}.{figure_format}", 
                   dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig_oor_distributions_top{top_n_clusters}.{figure_format}")

    print(f"[OOR Plots] All plots saved to {output_dir}/")

def plot_oor_pairwise(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    cluster_col: str = "Predicted_Label",
    condition_col: str = "Genotype",
    oor_col: str = "OOR_mahal_score",
    baseline_label: str = "WT",
    comparison_groups: List[str] = None,
    output_dir: str = "./oor",
    top_n_clusters: int = 10,
    figure_format: str = "png",
    figure_dpi: int = 150
):
    """Pairwise mode: All group combinations."""
    
    from itertools import combinations
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if comparison_groups is None:
        comparison_groups = df[condition_col].unique().tolist()
        comparison_groups = [g for g in comparison_groups if g != baseline_label]
    
    all_groups = [baseline_label] + comparison_groups
    
    # Color palette for ALL groups (including pairs)
    palette = sns.color_palette("Set2", len(all_groups))
    color_map = dict(zip(all_groups, palette))
    
    print(f"[OOR Plots] Pairwise mode: All combinations of {all_groups}")
    
    # ========================================================================
    # Plot 1: Median by cluster (ALL GROUPS)
    # ========================================================================
    if "Comparison" in summary_df.columns and "Baseline" in summary_df.columns:
        try:
            fig, ax = plt.subplots(figsize=(16, 6))
            
            clusters = summary_df['Cluster'].unique()
            x_pos = np.arange(len(clusters))
            
            # Collect all unique groups from summary
            all_unique_groups = list(set(
                list(summary_df['Baseline'].unique()) + 
                list(summary_df['Comparison'].unique())
            ))
            
            # Ensure color map has all groups
            if len(all_unique_groups) > len(color_map):
                extended_palette = sns.color_palette("Set2", len(all_unique_groups))
                color_map = dict(zip(all_unique_groups, extended_palette))
            
            # Plot each group (collect medians from both baseline and comparison)
            plotted_groups = {}
            
            for group in all_unique_groups:
                # Collect medians where this group appears as baseline
                baseline_data = summary_df[summary_df['Baseline'] == group]
                if not baseline_data.empty:
                    baseline_medians = baseline_data.groupby('Cluster')['median_baseline'].first()
                    for cluster in baseline_medians.index:
                        if group not in plotted_groups:
                            plotted_groups[group] = {}
                        plotted_groups[group][cluster] = baseline_medians[cluster]
                
                # Collect medians where this group appears as comparison
                comp_data = summary_df[summary_df['Comparison'] == group]
                if not comp_data.empty:
                    comp_medians = comp_data.groupby('Cluster')['median_comparison'].first()
                    for cluster in comp_medians.index:
                        if group not in plotted_groups:
                            plotted_groups[group] = {}
                        # Average if both exist, otherwise use whichever is available
                        if cluster in plotted_groups[group]:
                            plotted_groups[group][cluster] = (plotted_groups[group][cluster] + comp_medians[cluster]) / 2
                        else:
                            plotted_groups[group][cluster] = comp_medians[cluster]
            
            # Plot
            for group in sorted(plotted_groups.keys()):
                medians = [plotted_groups[group].get(cl, np.nan) for cl in clusters]
                if not all(np.isnan(medians)):
                    ax.scatter(x_pos, medians, label=group, 
                              color=color_map.get(group, '#999999'), 
                              s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(clusters, rotation=45, ha='right')
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel(f'Median {oor_col}', fontsize=12)
            ax.set_title('Median OOR: All Groups (Pairwise Mode)', fontsize=14)
            ax.legend(title=condition_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / f"fig_oor_median_by_cluster.{figure_format}", 
                       dpi=figure_dpi, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: fig_oor_median_by_cluster.{figure_format}")
        except Exception as e:
            print(f"  ⚠ Warning: Plot 1 failed: {e}")
    
    # ========================================================================
    # Plot 2: Pairwise comparison matrix (Fold Change)
    # ========================================================================
    try:
        comparison_pairs = []
        for _, row in summary_df.iterrows():
            pair_label = f"{row['Baseline']} vs {row['Comparison']}"
            comparison_pairs.append({
                'Cluster': row['Cluster'],
                'Pair': pair_label,
                'delta_median': row.get('delta_median', np.nan),
                'fold_change': row.get('fold_change', np.nan)
            })
        
        if comparison_pairs:
            pair_df = pd.DataFrame(comparison_pairs)
            
            # Heatmap of fold changes
            pivot = pair_df.pivot_table(
                index='Cluster',
                columns='Pair',
                values='fold_change',
                aggfunc='first'
            )
            
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8), 
                                               max(6, len(pivot) * 0.3 + 2)))
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', 
                           center=1, ax=ax, linewidths=0.5,
                           cbar_kws={'label': 'Fold Change'})
                ax.set_title('Pairwise Fold Change Matrix', fontsize=14)
                ax.set_xlabel('Comparison Pair', fontsize=12)
                ax.set_ylabel('Cluster', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(output_path / f"fig_oor_pairwise_matrix.{figure_format}", 
                           dpi=figure_dpi, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: fig_oor_pairwise_matrix.{figure_format}")
    except Exception as e:
        print(f"  ⚠ Warning: Pairwise matrix plot failed: {e}")
    
    # ========================================================================
    # Plot 3: Delta median comparison (all pairs)
    # ========================================================================
    try:
        if comparison_pairs:
            pair_df = pd.DataFrame(comparison_pairs)
            
            # Top N pairs by absolute delta
            pair_df['abs_delta'] = pair_df['delta_median'].abs()
            top_pairs = pair_df.nlargest(min(20, len(pair_df)), 'abs_delta')
            
            fig, ax = plt.subplots(figsize=(12, max(6, len(top_pairs) * 0.3)))
            
            colors_pair = ['#D32F2F' if x > 0 else '#2E7D32' for x in top_pairs['delta_median']]
            y_pos = np.arange(len(top_pairs))
            
            ax.barh(y_pos, top_pairs['delta_median'], 
                   color=colors_pair, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Create labels: Cluster (Pair)
            labels = [f"{row['Cluster']} ({row['Pair']})" for _, row in top_pairs.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('Δ Median OOR', fontsize=12)
            ax.set_title('Top Pairwise Differences', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / f"fig_oor_pairwise_delta.{figure_format}", 
                       dpi=figure_dpi, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: fig_oor_pairwise_delta.{figure_format}")
    except Exception as e:
        print(f"  ⚠ Warning: Delta plot failed: {e}")
    
    # ========================================================================
    # Plot 4: Network graph of comparisons (if not too many)
    # ========================================================================
    try:
        if len(all_groups) <= 5:  # Only for small number of groups
            # Create adjacency matrix
            n_groups = len(all_groups)
            adj_matrix = np.zeros((n_groups, n_groups))
            
            for _, row in summary_df.iterrows():
                try:
                    baseline_idx = all_groups.index(row['Baseline'])
                    comp_idx = all_groups.index(row['Comparison'])
                    # Use average delta_median across clusters as edge weight
                    adj_matrix[baseline_idx, comp_idx] = abs(row.get('delta_median', 0))
                except (ValueError, KeyError):
                    continue
            
            # Aggregate across all clusters
            comparison_strength = {}
            for i, g1 in enumerate(all_groups):
                for j, g2 in enumerate(all_groups):
                    if i != j and adj_matrix[i, j] > 0:
                        pair = f"{g1}-{g2}"
                        if pair not in comparison_strength:
                            comparison_strength[pair] = []
                        comparison_strength[pair].append(adj_matrix[i, j])
            
            # Average across clusters
            avg_strength = {k: np.mean(v) for k, v in comparison_strength.items()}
            
            if avg_strength:
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Circular layout
                angles = np.linspace(0, 2*np.pi, len(all_groups), endpoint=False)
                pos = {group: (np.cos(angle), np.sin(angle)) 
                       for group, angle in zip(all_groups, angles)}
                
                # Draw edges
                max_strength = max(avg_strength.values()) if avg_strength else 1
                for pair, strength in avg_strength.items():
                    g1, g2 = pair.split('-')
                    if g1 in pos and g2 in pos:
                        x1, y1 = pos[g1]
                        x2, y2 = pos[g2]
                        width = 1 + 4 * (strength / max_strength)
                        ax.plot([x1, x2], [y1, y2], 'gray', 
                               linewidth=width, alpha=0.5, zorder=1)
                
                # Draw nodes
                for group, (x, y) in pos.items():
                    ax.scatter(x, y, s=500, c=color_map.get(group, '#999999'), 
                              edgecolor='black', linewidth=2, zorder=2, alpha=0.8)
                    ax.text(x, y, group, ha='center', va='center', 
                           fontsize=12, fontweight='bold', zorder=3)
                
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title('Pairwise Comparison Network\n(Edge width = average Δ median)', 
                            fontsize=14, pad=20)
                
                plt.tight_layout()
                plt.savefig(output_path / f"fig_oor_pairwise_network.{figure_format}", 
                           dpi=figure_dpi, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: fig_oor_pairwise_network.{figure_format}")
    except Exception as e:
        print(f"  ⚠ Warning: Network plot failed: {e}")
    
    # ========================================================================
    # Plot 5: Distribution violin plots for top affected cluster
    # ========================================================================
    try:
        if top_n_clusters > 0 and not summary_df.empty:
            # Find top cluster with largest variation across all pairs
            cluster_variation = summary_df.groupby('Cluster')['delta_median'].apply(
                lambda x: x.abs().max()
            ).sort_values(ascending=False)
            
            top_cluster_list = cluster_variation.head(top_n_clusters).index.tolist()
            
            n_cols = min(3, len(top_cluster_list))
            n_rows = int(np.ceil(len(top_cluster_list) / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            
            for i, cluster in enumerate(top_cluster_list[:len(axes)]):
                ax = axes[i]
                # sub = df[df[cluster_col] == cluster]
                sub = df[df[cluster_col].astype(str).apply(lambda x: x.split('.')[0]) == str(cluster).split('.')[0]]
                
                # Violin plot
                sns.violinplot(data=sub, x=condition_col, y=oor_col, 
                              order=all_groups, palette=color_map, 
                              ax=ax, inner='quartile')
                
                # Add median lines
                medians = sub.groupby(condition_col)[oor_col].median()
                for j, group in enumerate(all_groups):
                    if group in medians.index:
                        ax.hlines(medians[group], j - 0.4, j + 0.4, 
                                colors='red', linewidth=2, linestyles='dashed')
                
                ax.set_title(f'{cluster}', fontsize=12)
                ax.set_xlabel('')
                ax.set_ylabel(oor_col if i % n_cols == 0 else '')
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle(f'Pairwise OOR Distribution: Top {top_n_clusters} Variable Clusters', 
                        fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(output_path / f"fig_oor_pairwise_distributions.{figure_format}", 
                       dpi=figure_dpi, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: fig_oor_pairwise_distributions.{figure_format}")
    except Exception as e:
        print(f"  ⚠ Warning: Distribution plot failed: {e}")
    
    print(f"[OOR Plots] Pairwise plots saved to {output_dir}/")


# ========================================================================
# Main dispatcher function
# ========================================================================
def generate_oor_plots(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_mode: str,
    **kwargs
):
    """
    Main entry point for OOR plotting.
    
    Dispatches to appropriate plotting function based on comparison_mode.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level OOR results
    summary_df : pd.DataFrame
        Cluster-level OOR summary
    comparison_mode : str
        One of: 'binary', 'multi', 'pairwise'
    **kwargs
        Additional arguments passed to plotting functions
    """
    
    mode = comparison_mode.lower().strip()
    
    print(f"\n[OOR Plots] Mode: {mode}")
    
    if mode == "binary":
        plot_oor_binary(df, summary_df, **kwargs)
    elif mode == "multi":
        plot_oor_multi_group(df, summary_df, **kwargs)
    elif mode == "pairwise":
        plot_oor_pairwise(df, summary_df, **kwargs)
    else:
        print(f"[OOR Plots] Warning: Unknown mode '{comparison_mode}', using 'multi' as fallback")
        plot_oor_multi_group(df, summary_df, **kwargs)