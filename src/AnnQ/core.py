import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from itertools import combinations

from scipy.stats import mannwhitneyu, ks_2samp
from scipy import stats
from .utils import get_combined_hierarchy



def calculate_entropy(probs):
    """Normalized Entropy"""
    probs_safe = np.where(probs <= 0, 1e-10, probs)
    n_cols = probs.shape[1]
    if n_cols <= 1: return np.zeros(len(probs))
    max_entropy = np.log(n_cols)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1) / max_entropy
    return entropy


def suggest_thresholds(prob_mat, verbose=True):
    """
    IQR 기반 adaptive threshold 제안
    
    P1 분포의 IQR을 사용하여 데이터에 맞는 threshold 자동 설정
    - Soft threshold: Q25 + 0.25*IQR (하위 25-50% 구간)
    - Hard threshold: Q50 + 0.5*IQR (중간 50-75% 구간)
    
    Args:
        prob_mat: Probability matrix (n_cells x n_celltypes)
        verbose: Print diagnostic information
    
    Returns:
        dict: Suggested thresholds and diagnostics
    """
    p1 = np.max(prob_mat, axis=1)
    p2 = np.partition(prob_mat, -2, axis=1)[:, -2]
    delta = p1 - p2
    
    # IQR 계산
    q25 = np.percentile(p1, 25)
    q50 = np.percentile(p1, 50)
    q75 = np.percentile(p1, 75)
    iqr = q75 - q25
    
    diagnostics = {
        'p1_q25': q25,
        'p1_q50': q50,
        'p1_q75': q75,
        'p1_iqr': iqr,
        'p1_mean': np.mean(p1),
        'delta_median': np.median(delta),
        'entropy_mean': calculate_entropy(prob_mat).mean(),
        'n_cells': len(prob_mat)
    }
    
    # IQR 기반 threshold 설정
    # Soft threshold: Q25와 Q50 사이 (하위 25-50%)
    soft_t = q25 + 0.25 * iqr
    
    # Hard threshold: Q50과 Q75 사이 (중간 50-75%)
    hard_t = q50 + 0.5 * iqr
    
    # Boundary 체크 (reasonable 범위로 제한)
    soft_t = np.clip(soft_t, 0.2, 0.5)
    hard_t = np.clip(hard_t, 0.4, 0.8)
    
    # Confidence level 판단
    if iqr > 0.4:
        confidence = 'high'
        rationale = f'Wide P1 distribution (IQR={iqr:.2f}) - clear separation between confident and ambiguous cells'
    elif iqr < 0.2:
        confidence = 'low'
        rationale = f'Narrow P1 distribution (IQR={iqr:.2f}) - most cells have similar confidence levels'
    else:
        confidence = 'medium'
        rationale = f'Moderate P1 distribution (IQR={iqr:.2f}) - typical heterogeneous population'
    
    suggestion = {
        'hard_threshold': round(hard_t, 2),
        'soft_threshold': round(soft_t, 2),
        'confidence': confidence,
        'rationale': rationale
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ADAPTIVE THRESHOLD SUGGESTION (IQR-based)")
        print("="*60)
        print(f"Data Confidence Level: {confidence.upper()}")
        print(f"\nP1 Distribution:")
        print(f"  Q25 (25th percentile): {q25:.3f}")
        print(f"  Q50 (median):          {q50:.3f}")
        print(f"  Q75 (75th percentile): {q75:.3f}")
        print(f"  IQR (Q75-Q25):         {iqr:.3f}")
        print(f"\nSuggested Thresholds:")
        print(f"  Soft threshold:  {suggestion['soft_threshold']:.2f} (Q25 + 0.25*IQR)")
        print(f"  Hard threshold:  {suggestion['hard_threshold']:.2f} (Q50 + 0.5*IQR)")
        print(f"\nRationale: {rationale}")
        print(f"\nAdditional Metrics:")
        print(f"  Mean P1:         {diagnostics['p1_mean']:.3f}")
        print(f"  Median Delta:    {diagnostics['delta_median']:.3f}")
        print(f"  Mean Entropy:    {diagnostics['entropy_mean']:.3f}")
        print("="*60 + "\n")
    
    suggestion['diagnostics'] = diagnostics
    return suggestion


def refine_g1_classification(meta_df, config):
    """
    G1을 biological meaning 기반으로 세분화
    
    G1 → G1a (LowConf), G1b (Ambiguous), G1c (Diffuse)
    
    Args:
        meta_df: DataFrame with classification results
        config: Configuration dict with refinement rules
    
    Returns:
        DataFrame with refined G1 classification
    """
    refine_g1 = config['parameters'].get('refine_g1', False)
    if not refine_g1:
        return meta_df
    
    g1_mask = meta_df['annq_group'] == 'G1'
    n_g1 = g1_mask.sum()
    
    if n_g1 == 0:
        print("No G1 cells to refine.")
        return meta_df
    
    print(f"\nRefining {n_g1} G1 cells into subgroups...")
    
    # Get refinement rules from config (with defaults)
    rules = config['parameters'].get('g1_refinement_rules', {})
    lowconf_p1_max = rules.get('lowconf_p1_max', 0.25)
    ambiguous_p1_min = rules.get('ambiguous_p1_min', 0.3)
    ambiguous_ratio_min = rules.get('ambiguous_ratio_min', 0.6)
    ambiguous_delta_max = rules.get('ambiguous_delta_max', 0.2)
    diffuse_entropy_min = rules.get('diffuse_entropy_min', 0.7)
    
    # Extract metrics for G1 cells
    p1 = meta_df.loc[g1_mask, 'A_Assurance'].values
    p2_p1_ratio = meta_df.loc[g1_mask, 'A_Admixture'].values
    delta = meta_df.loc[g1_mask, 'AQ_Delta'].values
    entropy = meta_df.loc[g1_mask, 'A_Ambiguity'].values
    
    # Initialize refined groups
    refined_groups = np.full(n_g1, 'G1', dtype=object)
    
    # G1a: True low confidence (everything is low)
    lowconf_mask = (p1 < lowconf_p1_max)
    refined_groups[lowconf_mask] = 'G1a_LowConf'
    
    # G1b: Ambiguous between 2 types (biological interest!)
    ambiguous_mask = (
        (p1 >= ambiguous_p1_min) & 
        (p2_p1_ratio >= ambiguous_ratio_min) & 
        (delta < ambiguous_delta_max)
    )
    refined_groups[ambiguous_mask] = 'G1b_Ambiguous'
    
    # G1c: Diffuse across many types
    diffuse_mask = (
        (p1 >= lowconf_p1_max) & 
        (entropy > diffuse_entropy_min) & 
        ~ambiguous_mask & 
        ~lowconf_mask
    )
    refined_groups[diffuse_mask] = 'G1c_Diffuse'
    
    # Update dataframe
    meta_df.loc[g1_mask, 'annq_group_refined'] = refined_groups
    
    # For non-G1 cells, copy the original group
    meta_df.loc[~g1_mask, 'annq_group_refined'] = meta_df.loc[~g1_mask, 'annq_group']
    
    print(f"\nG1 Refinement Results:")
    refine_counts = meta_df.loc[g1_mask, 'annq_group_refined'].value_counts()
    for group, count in refine_counts.items():
        pct = count / n_g1 * 100
        print(f"  {group}: {count} ({pct:.1f}%)")
    
    return meta_df


def clean_column_name(col_name):
    """
    컬럼명에서 cell type 이름만 추출합니다.
    
    지원하는 형식:
    - "prediction.score.Upper.Layer" -> "Upper.Layer"
    - 나머지는 모두 원본 유지 (JSON hierarchy 매칭용)
    """
    clean_name = col_name.strip()
    
    # prediction.score. 접두사만 제거 (Seurat/label transfer 형식)
    if clean_name.startswith("prediction.score."):
        clean_name = clean_name.replace("prediction.score.", "")
    
    # 나머지는 원본 그대로 반환
    return clean_name


def run_classification(config):
    start_time = time.time()
    
    # 1. Config 파싱 (구조 변경 반영)
    input_file = config['input']['file_path']
    data_start_idx = config['input']['start_col_index'] - 1
    
    # Classification mode
    mode = config['parameters'].get('mode', 'standard')
    threshold_hard = config['parameters'].get('threshold', 0.5)
    threshold_soft = config['parameters'].get('soft_threshold', None)
    auto_suggest = config['parameters'].get('auto_suggest', False)
    
    # Annotation 섹션 읽기
    anno_config = config.get('annotation', {})
    tool_name = anno_config.get('tool', 'celltypist')
    model_list = anno_config.get('models', [])
    hierarchy_file = anno_config.get('hierarchy_file', None)
    
    # 모델 리스트가 문자열 하나로만 들어왔을 경우 리스트로 변환
    if isinstance(model_list, str):
        model_list = [model_list]

    print(f"[{tool_name.upper()}] Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Shape: {df.shape}")
    print(f"  Classification Mode: {mode.upper()}")
    
    # 데이터 분리
    meta_df = df.iloc[:, :data_start_idx].copy()
    prob_df = df.iloc[:, data_start_idx:]
    prob_mat = prob_df.values
    cell_type_names = prob_df.columns.tolist()
    
    # NA 처리
    if np.isnan(prob_mat).any():
        prob_mat = np.nan_to_num(prob_mat, nan=0.0)

    # ---------------------------------------------------------
    # Auto-suggest thresholds (adaptive mode or when requested)
    # ---------------------------------------------------------
    if mode == 'adaptive' or auto_suggest:
        suggestion = suggest_thresholds(prob_mat, verbose=True)
        
        if mode == 'adaptive':
            threshold_hard = suggestion['hard_threshold']
            threshold_soft = suggestion['soft_threshold']
            print(f"Using adaptive thresholds: hard={threshold_hard}, soft={threshold_soft}")
        else:
            print("(Suggestion only - using config values)")
    
    # Set soft threshold if not specified
    if threshold_soft is None:
        if mode == 'soft':
            threshold_soft = 0.3  # default for soft mode
        else:
            threshold_soft = threshold_hard  # same as hard (no soft classification)
    
    print(f"Thresholds: hard={threshold_hard}, soft={threshold_soft}")

    # ---------------------------------------------------------
    # Hierarchy 로드
    # ---------------------------------------------------------
    hierarchy_map = {}
    
    if hierarchy_file:
        print(f"Loading custom hierarchy from: {hierarchy_file}")
        hierarchy_map = get_combined_hierarchy(model_list, hierarchy_file=hierarchy_file)
    elif model_list:
        print(f"Loading hierarchy for models: {', '.join(model_list)}")
        hierarchy_map = get_combined_hierarchy(model_list)
    else:
        print("Warning: No hierarchy specified. Classification will use defaults.")

    # ---------------------------------------------------------
    # Classification Logic
    # ---------------------------------------------------------
    print("Classifying cells...")
    
    # Calculate hit counts for both thresholds
    hard_hits = np.sum(prob_mat >= threshold_hard, axis=1)
    soft_hits = np.sum(prob_mat >= threshold_soft, axis=1)
    
    # Initialize groups
    groups = np.full(len(df), "G1", dtype=object)
    
    # G0: Single hard hit
    groups[hard_hits == 1] = "G0"
    
    # Identify complex cases
    if mode in ['soft', 'adaptive']:
        # Standard multi-hit (hard threshold)
        hard_multi_indices = np.where(hard_hits >= 2)[0]
        
        # Soft multi-hit (soft threshold but not hard)
        soft_multi_indices = np.where((hard_hits < 2) & (soft_hits >= 2))[0]
        
        print(f"  G0 (single clear hit): {(hard_hits == 1).sum()} cells")
        print(f"  Hard multi-hit (>= {threshold_hard}): {len(hard_multi_indices)} cells")
        print(f"  Soft multi-hit ({threshold_soft}-{threshold_hard}): {len(soft_multi_indices)} cells")
        
        # Process hard multi-hits
        if len(hard_multi_indices) > 0:
            groups = classify_multi_hits(
                groups, hard_multi_indices, prob_mat, cell_type_names,
                threshold_hard, hierarchy_map, suffix=""
            )
        
        # Process soft multi-hits
        if len(soft_multi_indices) > 0:
            groups = classify_multi_hits(
                groups, soft_multi_indices, prob_mat, cell_type_names,
                threshold_soft, hierarchy_map, suffix="s"
            )
    
    else:  # standard mode
        multi_hit_indices = np.where(hard_hits >= 2)[0]
        print(f"  Multi-hit cells: {len(multi_hit_indices)}")
        
        if len(multi_hit_indices) > 0:
            groups = classify_multi_hits(
                groups, multi_hit_indices, prob_mat, cell_type_names,
                threshold_hard, hierarchy_map, suffix=""
            )

    # ---------------------------------------------------------
    # QC Metrics
    # ---------------------------------------------------------
    print("Calculating QC Metrics...")
    sorted_indices = np.argsort(prob_mat, axis=1)
    p1 = prob_mat[np.arange(len(df)), sorted_indices[:, -1]]
    p2 = prob_mat[np.arange(len(df)), sorted_indices[:, -2]]
    
    aq_delta = p1 - p2
    aq_totalsum = np.sum(prob_mat * (prob_mat >= threshold_hard), axis=1)
    a_ambiguity = calculate_entropy(prob_mat)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        a_admixture = np.where(p1 > 0, p2 / p1, 0)
    
    best_celltypes = prob_df.columns[sorted_indices[:, -1]]

    # 결과 병합
    meta_df['best_celltype'] = best_celltypes
    meta_df['annq_group'] = groups
    meta_df['AQ_Delta'] = np.round(aq_delta, 4)
    meta_df['AQ_TotalSum'] = np.round(aq_totalsum, 4)
    meta_df['A_Assurance'] = np.round(p1, 4)
    meta_df['A_Ambiguity'] = np.round(a_ambiguity, 4)
    meta_df['A_Admixture'] = np.round(a_admixture, 4)
    meta_df['P1'] = np.round(p1, 4)  # Add P1
    meta_df['P2'] = np.round(p2, 4)  # Add P2
    
    # Tool & mode 정보
    meta_df['annotation_tool'] = tool_name
    meta_df['classification_mode'] = mode
    
    # ---------------------------------------------------------
    # G1 Refinement (optional)
    # ---------------------------------------------------------
    meta_df = refine_g1_classification(meta_df, config)
    
    # ---------------------------------------------------------
    # 저장
    # ---------------------------------------------------------
    output_dir = config['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if config['output']['save_csv']:
        output_path = os.path.join(output_dir, "classified_results.csv.gz")
        meta_df.to_csv(output_path, index=False, compression='gzip')
        print(f"Saved results to {output_path}")

    print(f"Total processing time: {time.time() - start_time:.2f}s")
    print("\nClassification Summary:")
    print(meta_df['annq_group'].value_counts().sort_index())
    
    if 'annq_group_refined' in meta_df.columns:
        print("\nRefined Classification Summary:")
        print(meta_df['annq_group_refined'].value_counts().sort_index())
    
    return meta_df


def _get_oor_config(config: dict) -> dict:
    oor_cfg = config.get("oor", {})
    
    if not oor_cfg:
        # No OOR config, return minimal defaults
        return {
            "enabled": False,
            "cell_id_col": "cell_id",
            "cluster_col": "Predicted_Label",
            "condition_col": "Genotype",
            "baseline_label": "WT",
            "comparison_mode": "binary",
            "comparison_groups": [],
            "feature_cols": ["P1", "P2", "Delta", "Entropy"],
            "min_baseline_cells_per_cluster": 30,
            "min_comparison_cells_per_cluster": 10,
            "tail_q": 0.95,
            "out_col": "OOR_mahal_score",
        }
    

    return {
        "enabled": oor_cfg.get("enabled", True),
        "cell_id_col": oor_cfg.get("cell_id_col", "cell_id"),
        "cluster_col": oor_cfg.get("cluster_col", "Predicted_Label"),
        "condition_col": oor_cfg.get("condition_col", "Genotype"),
        "baseline_label": oor_cfg.get("baseline_label", "WT"),
        "comparison_mode": oor_cfg.get("comparison_mode", "binary"),
        "comparison_groups": oor_cfg.get("comparison_groups", []),
        "feature_cols": oor_cfg.get("feature_cols", ["P1", "P2", "Delta", "Entropy"]),
        "min_baseline_cells_per_cluster": oor_cfg.get("min_baseline_cells_per_cluster", 30),
        "min_comparison_cells_per_cluster": oor_cfg.get("min_comparison_cells_per_cluster", 10),
        "tail_q": oor_cfg.get("tail_q", 0.95),
        "out_col": oor_cfg.get("out_col", "OOR_mahal_score"),
        "save_cell_scores": oor_cfg.get("save_cell_scores", True),
        "save_cluster_summary": oor_cfg.get("save_cluster_summary", True),
        "oor_subdir": oor_cfg.get("oor_subdir", "oor"),
        "generate_plots": oor_cfg.get("generate_plots", True),
        "plot_settings": oor_cfg.get("plot_settings", {}),
    }


def _mahalanobis_scores(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute squared Mahalanobis distance for each row in X."""
    # Regularize covariance for stability
    d = cov.shape[0]
    cov_reg = cov + np.eye(d) * 1e-8
    inv_cov = np.linalg.pinv(cov_reg)
    diff = X - mu
    # (diff @ inv_cov) * diff row-wise sum
    return np.einsum('ij,jk,ik->i', diff, inv_cov, diff)


def compute_oor_by_cluster(
    df: pd.DataFrame,
    *,
    cell_id_col: str,
    cluster_col: str,
    condition_col: str,
    feature_cols: List[str],
    baseline_label: str,
    comparison_mode: str = "multi",
    comparison_groups: Optional[List[str]] = None,
    min_baseline_cells_per_cluster: int = 30,
    min_comparison_cells_per_cluster: int = 10,
    tail_q: float = 0.95,
    out_col: str = "OOR_mahal_score",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced OOR analysis with flexible multi-group comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with classification results
    cell_id_col : str
        Column name for cell identifiers
    cluster_col : str
        Column name for cluster/cell type annotations
    condition_col : str
        Column name for experimental conditions (e.g., Genotype)
    feature_cols : List[str]
        Features for Mahalanobis distance (e.g., ['P1', 'P2', 'Delta', 'Entropy'])
    baseline_label : str
        Label for baseline group (e.g., 'WT')
    comparison_mode : str, default='multi'
        - 'binary': baseline vs all others combined
        - 'multi': baseline vs each group separately
        - 'pairwise': all pairwise comparisons
    comparison_groups : Optional[List[str]], default=None
        Explicit list of comparison groups. If None, auto-detect from data.
    min_baseline_cells_per_cluster : int, default=30
        Minimum baseline cells required per cluster
    min_comparison_cells_per_cluster : int, default=10
        Minimum comparison cells required per cluster
    tail_q : float, default=0.95
        Quantile threshold for tail enrichment
    out_col : str, default='OOR_mahal_score'
        Output column name for OOR scores
        
    Returns
    -------
    work : pd.DataFrame
        Input dataframe with added OOR score column
    summary_df : pd.DataFrame
        Per-cluster summary statistics
    """
    work = df.copy()

    # Validate columns
    for c in [cell_id_col, cluster_col, condition_col]:
        if c not in work.columns:
            raise KeyError(f"Missing required column: '{c}'")
    missing_feats = [c for c in feature_cols if c not in work.columns]
    if missing_feats:
        raise KeyError(f"Missing OOR feature columns: {missing_feats}")

    # Ensure numeric features
    X_all = work[feature_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    if np.isnan(X_all).any():
        col_medians = np.nanmedian(X_all, axis=0)
        inds = np.where(np.isnan(X_all))
        X_all[inds] = np.take(col_medians, inds[1])
    work[feature_cols] = X_all

    # Auto-detect comparison groups if not specified
    if comparison_groups is None or len(comparison_groups) == 0:
        all_groups = work[condition_col].unique().tolist()
        comparison_groups = [g for g in all_groups if g != baseline_label]
    
    print(f"[OOR] Baseline: {baseline_label}")
    print(f"[OOR] Comparison groups: {comparison_groups}")
    print(f"[OOR] Comparison mode: {comparison_mode}")

    # Global baseline (fallback)
    baseline_mask_global = (work[condition_col] == baseline_label)
    if baseline_mask_global.sum() < max(min_baseline_cells_per_cluster, len(feature_cols) + 5):
        raise ValueError(
            f"Not enough baseline cells: baseline_label='{baseline_label}', count={baseline_mask_global.sum()}"
        )
    X_baseline_global = work.loc[baseline_mask_global, feature_cols].to_numpy()
    mu_global = X_baseline_global.mean(axis=0)
    cov_global = np.cov(X_baseline_global, rowvar=False)

    # Allocate score column
    scores = np.full(len(work), np.nan, dtype=float)

    # Cluster-wise scoring
    clusters = work[cluster_col].astype(str).unique()
    
    for cl in clusters:
        idx = (work[cluster_col].astype(str) == cl)
        baseline_idx = idx & baseline_mask_global
        X = work.loc[idx, feature_cols].to_numpy()

        # Use cluster-specific baseline if sufficient cells
        if baseline_idx.sum() >= max(min_baseline_cells_per_cluster, len(feature_cols) + 5):
            X_baseline = work.loc[baseline_idx, feature_cols].to_numpy()
            mu = X_baseline.mean(axis=0)
            cov = np.cov(X_baseline, rowvar=False)
        else:
            mu = mu_global
            cov = cov_global

        scores[idx.to_numpy()] = _mahalanobis_scores(X, mu, cov)

    work[out_col] = scores

    # ========================================================================
    # Per-cluster summary
    # ========================================================================
    summaries = []
    
    for cl in clusters:
        sub = work[work[cluster_col].astype(str) == cl]
        sub_baseline = sub[sub[condition_col] == baseline_label]
        
        if len(sub_baseline) < min_baseline_cells_per_cluster:
            continue
        
        # Baseline statistics
        baseline_vals = sub_baseline[out_col].to_numpy()
        tail_thr = float(np.quantile(baseline_vals, tail_q))
        baseline_median = float(np.median(baseline_vals))
        baseline_tail_frac = float((baseline_vals > tail_thr).mean())
        
        # Mode-specific comparisons
        if comparison_mode == "binary":
            # Baseline vs all others combined
            sub_comparison = sub[sub[condition_col] != baseline_label]
            if len(sub_comparison) < min_comparison_cells_per_cluster:
                continue
                
            summaries.append(_compute_comparison_stats(
                cluster=cl,
                baseline_vals=baseline_vals,
                comparison_vals=sub_comparison[out_col].to_numpy(),
                baseline_label=baseline_label,
                comparison_label="NonBaseline",
                n_baseline=len(sub_baseline),
                n_comparison=len(sub_comparison),
                baseline_median=baseline_median,
                tail_thr=tail_thr,
                baseline_tail_frac=baseline_tail_frac,
                tail_q=tail_q
            ))
            
        elif comparison_mode == "multi":
            # Baseline vs each group separately
            for group in comparison_groups:
                sub_group = sub[sub[condition_col] == group]
                if len(sub_group) < min_comparison_cells_per_cluster:
                    continue
                    
                summaries.append(_compute_comparison_stats(
                    cluster=cl,
                    baseline_vals=baseline_vals,
                    comparison_vals=sub_group[out_col].to_numpy(),
                    baseline_label=baseline_label,
                    comparison_label=group,
                    n_baseline=len(sub_baseline),
                    n_comparison=len(sub_group),
                    baseline_median=baseline_median,
                    tail_thr=tail_thr,
                    baseline_tail_frac=baseline_tail_frac,
                    tail_q=tail_q
                ))
        
        elif comparison_mode == "pairwise":
            # All pairwise comparisons (including between comparison groups)
            all_groups = [baseline_label] + comparison_groups
            for g1, g2 in combinations(all_groups, 2):
                sub_g1 = sub[sub[condition_col] == g1]
                sub_g2 = sub[sub[condition_col] == g2]
                
                min_cells = min_comparison_cells_per_cluster if g1 != baseline_label and g2 != baseline_label else min_baseline_cells_per_cluster
                
                if len(sub_g1) < min_cells or len(sub_g2) < min_cells:
                    continue
                
                # Use g1 as reference for this pair
                g1_vals = sub_g1[out_col].to_numpy()
                g1_median = float(np.median(g1_vals))
                g1_tail_thr = float(np.quantile(g1_vals, tail_q))
                g1_tail_frac = float((g1_vals > g1_tail_thr).mean())
                
                summaries.append(_compute_comparison_stats(
                    cluster=cl,
                    baseline_vals=g1_vals,
                    comparison_vals=sub_g2[out_col].to_numpy(),
                    baseline_label=g1,
                    comparison_label=g2,
                    n_baseline=len(sub_g1),
                    n_comparison=len(sub_g2),
                    baseline_median=g1_median,
                    tail_thr=g1_tail_thr,
                    baseline_tail_frac=g1_tail_frac,
                    tail_q=tail_q
                ))

    summary_df = pd.DataFrame(summaries)
    
    if not summary_df.empty:
        # Sort by effect size
        if comparison_mode in ["binary", "multi"]:
            summary_df = summary_df.sort_values(["delta_median", "tail_frac_comparison"], ascending=[False, False])
        else:
            summary_df = summary_df.sort_values(["Comparison", "delta_median"], ascending=[True, False])
    
    return work, summary_df

def _compute_comparison_stats(
    cluster: str,
    baseline_vals: np.ndarray,
    comparison_vals: np.ndarray,
    baseline_label: str,
    comparison_label: str,
    n_baseline: int,
    n_comparison: int,
    baseline_median: float,
    tail_thr: float,
    baseline_tail_frac: float,
    tail_q: float
) -> Dict:
    """Helper function to compute statistics for a single comparison."""
    
    comparison_median = float(np.median(comparison_vals))
    delta_median = comparison_median - baseline_median
    fold_change = comparison_median / baseline_median if baseline_median > 0 else np.inf
    
    comparison_tail_frac = float((comparison_vals > tail_thr).mean())
    tail_enrichment = comparison_tail_frac / baseline_tail_frac if baseline_tail_frac > 0 else np.inf
    
    # Statistical tests
    try:
        mw_p = mannwhitneyu(baseline_vals, comparison_vals, alternative='two-sided').pvalue
    except Exception:
        mw_p = np.nan
    
    try:
        ks_p = ks_2samp(baseline_vals, comparison_vals, alternative='two-sided').pvalue
    except Exception:
        ks_p = np.nan
    
    return {
        "Cluster": cluster,
        "Baseline": baseline_label,
        "Comparison": comparison_label,
        "n_baseline": int(n_baseline),
        "n_comparison": int(n_comparison),
        "median_baseline": baseline_median,
        "median_comparison": comparison_median,
        "delta_median": delta_median,
        "fold_change": fold_change,
        "tail_q": tail_q,
        "tail_threshold": tail_thr,
        "tail_frac_baseline": baseline_tail_frac,
        "tail_frac_comparison": comparison_tail_frac,
        "tail_enrichment": tail_enrichment,
        "p_mannwhitney": float(mw_p) if not np.isnan(mw_p) else np.nan,
        "p_ks": float(ks_p) if not np.isnan(ks_p) else np.nan,
    }


def run_oor_analysis(config: dict) -> None:
    """
    Run post-hoc OOR analysis using AnnQ's saved results.
    v3.4: Supports binary, multi, and pairwise comparison modes.
    """
    from pathlib import Path
    
    # Get output directory
    base_outdir = config["output"]["dir"]
    
    # Get OOR config
    oor_cfg = _get_oor_config(config)
    # Extract variables for convenience
    cluster_col = oor_cfg["cluster_col"]
    condition_col = oor_cfg["condition_col"]
    baseline_label = oor_cfg["baseline_label"]
    comparison_mode = oor_cfg["comparison_mode"]
    comparison_groups = oor_cfg["comparison_groups"]
    out_col = oor_cfg["out_col"]
    
    # Setup output directory
    oor_subdir = oor_cfg["oor_subdir"]
    output_dir = os.path.join(base_outdir, oor_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load classified results
    in_path = os.path.join(base_outdir, "classified_results.csv.gz")
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Cannot find '{in_path}'. "
            "Run 'annq -c <config.yaml>' first to generate classified_results.csv.gz"
        )

    print(f"\n{'='*70}")
    print("OOR ANALYSIS v3.4")
    print(f"{'='*70}")
    print(f"Loading AnnQ results: {in_path}")
    df = pd.read_csv(in_path, compression='gzip', low_memory=False)
    print(f"  Shape: {df.shape}")

    # Display config
    print(f"\nOOR Configuration:")
    print(f"  Condition column: {oor_cfg['condition_col']}")
    print(f"  Cluster column:   {oor_cfg['cluster_col']}")
    print(f"  Baseline label:   {oor_cfg['baseline_label']}")
    print(f"  Comparison mode:  {oor_cfg['comparison_mode']}")
    print(f"  Features:         {oor_cfg['feature_cols']}")
    print(f"  Tail quantile:    {oor_cfg['tail_q']}")

    # Run OOR calculation
    scored_df, summary_df = compute_oor_by_cluster(
        df,
        cell_id_col=oor_cfg["cell_id_col"],
        cluster_col=oor_cfg["cluster_col"],
        condition_col=oor_cfg["condition_col"],
        feature_cols=oor_cfg["feature_cols"],
        baseline_label=oor_cfg["baseline_label"],
        comparison_mode=oor_cfg["comparison_mode"],
        comparison_groups=oor_cfg["comparison_groups"],
        min_baseline_cells_per_cluster=oor_cfg["min_baseline_cells_per_cluster"],
        min_comparison_cells_per_cluster=oor_cfg["min_comparison_cells_per_cluster"],
        tail_q=oor_cfg["tail_q"],
        out_col=oor_cfg["out_col"],
    )

    # Save results
    if oor_cfg["save_cell_scores"]:
        out_scored = os.path.join(output_dir, "oor_results.csv.gz")
        scored_df.to_csv(out_scored, index=False, compression='gzip')
        print(f"\n[OOR] Saved cell-level scores: {out_scored}")
    
    if oor_cfg["save_cluster_summary"]:
        out_summary = os.path.join(output_dir, "oor_cluster_summary.csv")
        summary_df.to_csv(out_summary, index=False)
        print(f"[OOR] Saved cluster summary: {out_summary}")

    # Generate plots
    if oor_cfg["generate_plots"] and not summary_df.empty:
        try:
            from .plotting import generate_oor_plots
            
            print(f"\n[OOR] Generating plots...")
            
            generate_oor_plots(
                df=scored_df,
                summary_df=summary_df,
                comparison_mode=comparison_mode,  # "binary" | "multi" | "pairwise"
                cluster_col=cluster_col,
                condition_col=condition_col,
                oor_col=out_col,
                baseline_label=baseline_label,
                comparison_groups=comparison_groups,
                output_dir=output_dir,
                top_n_clusters=10,
                figure_format="png",
                figure_dpi=150
            )
            
            #print(f"[OOR] Plots saved to: {output_dir}/")
            
        except ImportError as e:
            print(f"\n[OOR] Warning: Could not import oor_plotting module: {e}")
            print("[OOR] Skipping plot generation.")
        except Exception as e:
            print(f"\n[OOR] Warning: Plot generation failed: {e}")
            print("[OOR] Results saved, but plots were not generated.")
    
    # Print summary
    if not summary_df.empty:
        print(f"\n{'='*70}")
        print("TOP 10 AFFECTED CLUSTERS")
        print(f"{'='*70}")
        
        top10 = summary_df.nlargest(10, 'delta_median')
        
        # Select columns based on comparison mode
        if oor_cfg["comparison_mode"] == "binary":
            cols_to_show = ['Cluster', 'n_baseline', 'n_comparison', 
                          'median_baseline', 'median_comparison', 'delta_median', 
                          'tail_enrichment', 'p_mannwhitney']
        else:
            cols_to_show = ['Cluster', 'Baseline', 'Comparison', 
                          'median_baseline', 'median_comparison', 'delta_median',
                          'fold_change', 'tail_enrichment', 'p_mannwhitney']
        
        print(top10[cols_to_show].to_string(index=False))
        print(f"{'='*70}\n")


def classify_multi_hits(groups, indices, prob_mat, cell_type_names, threshold, 
                        hierarchy_map, suffix=""):
    """
    Helper function to classify multi-hit cells using hierarchy
    
    Args:
        groups: Array of group labels
        indices: Indices of cells to classify
        prob_mat: Probability matrix
        cell_type_names: List of cell type names
        threshold: Probability threshold to use
        hierarchy_map: Cell type hierarchy mapping
        suffix: Suffix for group labels (e.g., "s" for soft hits)
    
    Returns:
        Updated groups array
    """
    # 매핑 테이블 생성
    col_info = []
    unmatched_types = set()
    
    for col in cell_type_names:
        clean_name = clean_column_name(col)
        
        if clean_name in hierarchy_map:
            col_info.append(hierarchy_map[clean_name])
        else:
            col_info.append({"Tissue": "Unknown", "Cell_Class": "Unknown"})
            unmatched_types.add(clean_name)
    
    if unmatched_types and not suffix:  # Only warn once
        print(f"  Warning: {len(unmatched_types)} cell types not found in hierarchy")
        for ct in list(unmatched_types)[:3]:
            print(f"    - {ct}")
        if len(unmatched_types) > 3:
            print(f"    ... and {len(unmatched_types) - 3} more")
    
    col_info_arr = np.array(col_info)

    # Process each multi-hit cell
    for row_idx in indices:
        row_probs = prob_mat[row_idx]
        hit_col_indices = np.where(row_probs >= threshold)[0]
        
        hit_infos = col_info_arr[hit_col_indices]
        
        tissues = set(info['Tissue'] for info in hit_infos)
        classes = set(info['Cell_Class'] for info in hit_infos)
        
        if len(tissues) > 1:
            # Previously labeled as G4 (multi-hit across different tissues). We fold this into G3 to avoid using G4.
            groups[row_idx] = f"G3{suffix}"
        elif len(classes) > 1:
            groups[row_idx] = f"G3{suffix}"
        else:
            groups[row_idx] = f"G2{suffix}"
    
    return groups
