"""
AnnQ - Annotation Quality Control & Classification
Enhanced cell type classification system for single-cell RNA-seq data
"""

__version__ = "3.4.0"
__author__ = "Joon An"
__email__ = "joon.an@korea.ac.kr"

from .core import run_classification, calculate_entropy
from .plotting import plot_results

__all__ = [
    "run_classification",
    "calculate_entropy",
    "plot_results",
]
