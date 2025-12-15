#!/usr/bin/env python
"""
Advanced visualization utilities for benchmark results.

This module provides publication-quality charts and interactive visualizations
for benchmark comparisons.
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BenchmarkVisualizer:
    """Generate publication-quality visualizations for benchmarks."""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer with style.
        
        Args:
            style: Matplotlib style to use
        """
        available_styles = plt.style.available
        if style in available_styles:
            plt.style.use(style)
        
        self.colors = {
            "Neural DSL": "#FF6B6B",
            "Keras": "#4ECDC4",
            "Raw TensorFlow": "#95E1D3",
            "PyTorch Lightning": "#FFD93D",
            "Raw PyTorch": "#F8B500",
            "Fast.ai": "#A8E6CF",
            "Ludwig": "#C7CEEA",
        }
    
    def load_results(self, results_path: str) -> pd.DataFrame:
        """Load benchmark results from JSON file."""
        with open(results_path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    def plot_metric_comparison(
        self,
        df: pd.DataFrame,
        metric: str,
        title: str,
        ylabel: str,
        lower_is_better: bool = True,
        output_path: Optional[Path] = None,
    ):
        """
        Create a bar chart comparing frameworks on a single metric.
        
        Args:
            df: DataFrame with benchmark results
            metric: Column name to plot
            title: Chart title
            ylabel: Y-axis label
            lower_is_better: Whether lower values are better
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frameworks = df["framework"].unique()
        values = [df[df["framework"] == fw][metric].mean() for fw in frameworks]
        
        colors = [self.colors.get(fw, "#CCCCCC") for fw in frameworks]
        
        bars = ax.bar(frameworks, values, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=1.5)
        
        ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Framework", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            format_str = '.2f' if val < 100 else '.0f'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:{format_str}}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        if lower_is_better:
            best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
        else:
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig, ax
    
    def plot_speedup_comparison(
        self,
        df: pd.DataFrame,
        baseline: str = "Neural DSL",
        output_path: Optional[Path] = None,
    ):
        """
        Create a speedup comparison chart showing relative performance.
        
        Args:
            df: DataFrame with benchmark results
            baseline: Framework to use as baseline (1.0x)
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frameworks = df["framework"].unique()
        baseline_time = df[df["framework"] == baseline]["development_time_seconds"].mean()
        
        speedups = []
        for fw in frameworks:
            fw_time = df[df["framework"] == fw]["development_time_seconds"].mean()
            speedup = baseline_time / fw_time if baseline_time > 0 else 1.0
            speedups.append(speedup)
        
        colors = [self.colors.get(fw, "#CCCCCC") for fw in frameworks]
        bars = ax.bar(frameworks, speedups, color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  label=f'{baseline} (baseline)')
        
        ax.set_title(f"Development Speed Relative to {baseline}", 
                    fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Framework", fontsize=14, fontweight="bold")
        ax.set_ylabel("Relative Speed (higher is faster)", fontsize=14, fontweight="bold")
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}x',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig, ax
    
    def plot_radar_chart(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        normalize: bool = True,
        output_path: Optional[Path] = None,
    ):
        """
        Create a radar chart comparing frameworks across multiple metrics.
        
        Args:
            df: DataFrame with benchmark results
            metrics: List of metrics to include
            normalize: Whether to normalize metrics to 0-1 scale
            output_path: Path to save figure
        """
        frameworks = df["framework"].unique()
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for fw in frameworks:
            fw_data = df[df["framework"] == fw]
            values = []
            
            for metric in metrics:
                value = fw_data[metric].mean()
                if normalize:
                    metric_min = df[metric].min()
                    metric_max = df[metric].max()
                    value = (value - metric_min) / (metric_max - metric_min) if metric_max != metric_min else 0.5
                values.append(value)
            
            values += values[:1]
            
            color = self.colors.get(fw, "#CCCCCC")
            ax.plot(angles, values, 'o-', linewidth=2, label=fw, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1 if normalize else None)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title("Multi-Metric Framework Comparison", 
                    fontsize=16, fontweight="bold", pad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig, ax
    
    def plot_heatmap(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        output_path: Optional[Path] = None,
    ):
        """
        Create a heatmap showing framework performance across metrics.
        
        Args:
            df: DataFrame with benchmark results
            metrics: List of metrics to include
            output_path: Path to save figure
        """
        frameworks = df["framework"].unique()
        
        data = []
        for fw in frameworks:
            fw_data = df[df["framework"] == fw]
            row = [fw_data[metric].mean() for metric in metrics]
            data.append(row)
        
        data_array = np.array(data)
        normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(normalized, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(frameworks)))
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_yticklabels(frameworks, fontsize=11)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(frameworks)):
            for j in range(len(metrics)):
                ax.text(j, i, f'{data_array[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight="bold")
        
        ax.set_title("Framework Performance Heatmap (normalized)", 
                    fontsize=18, fontweight="bold", pad=20)
        fig.colorbar(im, ax=ax, label="Normalized Score")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig, ax
    
    def plot_code_reduction(
        self,
        df: pd.DataFrame,
        baseline: str = "Raw PyTorch",
        output_path: Optional[Path] = None,
    ):
        """
        Create a visualization showing code reduction percentages.
        
        Args:
            df: DataFrame with benchmark results
            baseline: Framework to compare against
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frameworks = [fw for fw in df["framework"].unique() if fw != baseline]
        baseline_loc = df[df["framework"] == baseline]["lines_of_code"].mean()
        
        reductions = []
        for fw in frameworks:
            fw_loc = df[df["framework"] == fw]["lines_of_code"].mean()
            reduction = ((baseline_loc - fw_loc) / baseline_loc) * 100
            reductions.append(reduction)
        
        colors = [self.colors.get(fw, "#CCCCCC") for fw in frameworks]
        bars = ax.barh(frameworks, reductions, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.5)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title(f"Code Reduction vs. {baseline}", 
                    fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Code Reduction (%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Framework", fontsize=14, fontweight="bold")
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, reduction in zip(bars, reductions):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                   f'{reduction:.1f}%',
                   ha='left', va='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return fig, ax
    
    def generate_all_plots(
        self,
        results_path: str,
        output_dir: Path,
    ):
        """
        Generate all standard plots for benchmark results.
        
        Args:
            results_path: Path to JSON results file
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.load_results(results_path)
        
        print("Generating visualizations...")
        
        self.plot_metric_comparison(
            df, "lines_of_code", 
            "Lines of Code Comparison",
            "Lines of Code",
            lower_is_better=True,
            output_path=output_dir / "loc_comparison.png"
        )
        print("  ✓ Lines of code comparison")
        
        self.plot_metric_comparison(
            df, "training_time_seconds",
            "Training Time Comparison",
            "Training Time (seconds)",
            lower_is_better=True,
            output_path=output_dir / "training_time.png"
        )
        print("  ✓ Training time comparison")
        
        self.plot_metric_comparison(
            df, "inference_time_ms",
            "Inference Latency Comparison",
            "Inference Time (milliseconds)",
            lower_is_better=True,
            output_path=output_dir / "inference_time.png"
        )
        print("  ✓ Inference time comparison")
        
        self.plot_metric_comparison(
            df, "model_accuracy",
            "Model Accuracy Comparison",
            "Accuracy",
            lower_is_better=False,
            output_path=output_dir / "accuracy_comparison.png"
        )
        print("  ✓ Accuracy comparison")
        
        self.plot_speedup_comparison(
            df,
            baseline="Neural DSL",
            output_path=output_dir / "speedup_comparison.png"
        )
        print("  ✓ Speedup comparison")
        
        self.plot_code_reduction(
            df,
            baseline="Raw PyTorch",
            output_path=output_dir / "code_reduction.png"
        )
        print("  ✓ Code reduction")
        
        metrics = ["lines_of_code", "training_time_seconds", "model_accuracy", "inference_time_ms"]
        if all(m in df.columns for m in metrics):
            self.plot_radar_chart(
                df, metrics,
                normalize=True,
                output_path=output_dir / "radar_chart.png"
            )
            print("  ✓ Radar chart")
            
            self.plot_heatmap(
                df, metrics,
                output_path=output_dir / "heatmap.png"
            )
            print("  ✓ Heatmap")
        
        plt.close('all')
        
        print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    """CLI for visualization generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate benchmark visualizations"
    )
    parser.add_argument(
        "results_file",
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_visualizations",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_plots(args.results_file, Path(args.output_dir))


if __name__ == "__main__":
    main()
