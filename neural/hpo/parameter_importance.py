"""
Parameter importance analysis for hyperparameter optimization.
"""

import logging
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RandomForestRegressor = None
    LabelEncoder = None
    StandardScaler = None


# Configure logger
logger = logging.getLogger(__name__)

class ParameterImportanceAnalyzer:
    """Analyzes the importance of hyperparameters in optimization trials."""

    def __init__(self):
        """Initialize the parameter importance analyzer."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for parameter importance analysis. Install with: pip install scikit-learn")
        self.encoders = {}
        self.scaler = StandardScaler()
        self.model = None

    def analyze(self, trials: List[Dict[str, Any]], target_metric: str = 'score') -> Dict[str, float]:
        """
        Analyze parameter importance from a list of trials.

        Args:
            trials: List of trial dictionaries with parameters and scores
            target_metric: The metric to use for importance analysis

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not trials:
            logger.warning("No trials provided for parameter importance analysis")
            return {}

        # Extract parameters and target values
        X, y, param_names = self._prepare_data(trials, target_metric)

        if X.shape[0] < 5:
            logger.warning("Too few trials for reliable parameter importance analysis")
            return {param: 1.0 / len(param_names) for param in param_names}

        # Train a random forest model to estimate parameter importance
        try:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)

            # Get feature importances
            importances = self.model.feature_importances_

            # Create a dictionary mapping parameter names to importance scores
            importance_dict = {param: float(imp) for param, imp in zip(param_names, importances)}

            return importance_dict

        except Exception as e:
            logger.error(f"Error in parameter importance analysis: {str(e)}")
            return {param: 1.0 / len(param_names) for param in param_names}

    def _prepare_data(self, trials: List[Dict[str, Any]], target_metric: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for parameter importance analysis.

        Args:
            trials: List of trial dictionaries
            target_metric: The metric to use for importance analysis

        Returns:
            X: Feature matrix
            y: Target values
            param_names: List of parameter names
        """
        # Extract all parameter names from trials
        all_params = set()
        for trial in trials:
            all_params.update(trial.get('parameters', {}).keys())
        param_names = sorted(list(all_params))

        # Create feature matrix and target vector
        X_data = []
        y_data = []

        for trial in trials:
            params = trial.get('parameters', {})
            score = trial.get(target_metric, trial.get('score', None))

            if score is None:
                continue

            # Extract parameter values
            row = []
            for param in param_names:
                value = params.get(param, None)
                row.append(value)

            X_data.append(row)
            y_data.append(score)

        # Convert to numpy arrays
        X_raw = np.array(X_data)
        y = np.array(y_data)

        # Handle categorical parameters and missing values
        X = self._encode_and_scale(X_raw, param_names)

        return X, y, param_names

    def _encode_and_scale(self, X_raw: np.ndarray, param_names: List[str]) -> np.ndarray:
        """
        Encode categorical parameters and scale numerical parameters.

        Args:
            X_raw: Raw feature matrix
            param_names: List of parameter names

        Returns:
            Encoded and scaled feature matrix
        """
        X_encoded = np.zeros_like(X_raw, dtype=float)

        # Encode each column
        for i, param in enumerate(param_names):
            column = X_raw[:, i]

            # Check if column contains non-numeric values
            try:
                X_encoded[:, i] = column.astype(float)
            except (ValueError, TypeError):
                # Categorical parameter, use label encoding
                if param not in self.encoders:
                    self.encoders[param] = LabelEncoder()
                    # Fit the encoder on non-None values
                    non_none_values = [v for v in column if v is not None]
                    if non_none_values:
                        self.encoders[param].fit(non_none_values)

                # Replace None with a placeholder value
                column_with_placeholder = np.array([v if v is not None else 'NONE_PLACEHOLDER' for v in column])

                # Ensure all values are in the encoder's classes
                for value in np.unique(column_with_placeholder):
                    if value != 'NONE_PLACEHOLDER' and value not in self.encoders[param].classes_:
                        # Refit the encoder with the new value
                        self.encoders[param].fit(np.append(self.encoders[param].classes_, [value]))

                # Transform the column
                try:
                    X_encoded[:, i] = self.encoders[param].transform(column_with_placeholder)
                except Exception as e:
                    logger.error(f"Error encoding parameter {param}: {str(e)}")
                    # Use zeros as a fallback
                    X_encoded[:, i] = 0

        # Replace NaN values with column means
        for i in range(X_encoded.shape[1]):
            col = X_encoded[:, i]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                col[nan_mask] = np.nanmean(col) if np.any(~nan_mask) else 0
                X_encoded[:, i] = col

        # Scale the features
        try:
            X_scaled = self.scaler.fit_transform(X_encoded)
            return X_scaled
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return X_encoded

    def plot_importance(self, importance_dict: Dict[str, float],
                       title: str = "Hyperparameter Importance",
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot parameter importance as a bar chart.

        Args:
            importance_dict: Dictionary mapping parameter names to importance scores
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Sort parameters by importance
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        params = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bars
        y_pos = np.arange(len(params))
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_parallel_coordinates(self, trials: List[Dict[str, Any]],
                                 target_metric: str = 'score',
                                 top_n: int = 10,
                                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create a parallel coordinates plot of hyperparameters.

        Args:
            trials: List of trial dictionaries
            target_metric: The metric to use for coloring
            top_n: Number of top trials to highlight
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not trials:
            logger.warning("No trials provided for parallel coordinates plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
            return fig

        # Convert trials to DataFrame
        df_list = []
        for trial in trials:
            row = trial.get('parameters', {}).copy()
            row[target_metric] = trial.get(target_metric, trial.get('score', 0))
            df_list.append(row)

        df = pd.DataFrame(df_list)

        if df.empty or len(df.columns) <= 1:
            logger.warning("Insufficient data for parallel coordinates plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            return fig

        # Sort by target metric and mark top_n trials
        df = df.sort_values(by=target_metric, ascending=False)
        df['is_top'] = False
        df.iloc[:top_n, df.columns.get_loc('is_top')] = True

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use Seaborn's parallel coordinates plot
        # First, normalize all columns for better visualization
        cols_to_plot = [col for col in df.columns if col not in [target_metric, 'is_top']]

        if not cols_to_plot:
            logger.warning("No parameters to plot")
            ax.text(0.5, 0.5, "No parameters to plot", ha='center', va='center')
            return fig

        df_norm = df.copy()
        for col in cols_to_plot:
            if df[col].dtype in [np.float64, np.int64]:
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

        # Plot non-top trials with low alpha
        for i, (_, row) in enumerate(df_norm[~df['is_top']].iterrows()):
            ax.plot(cols_to_plot, row[cols_to_plot], color='gray', alpha=0.3)

        # Plot top trials with high alpha and different color
        for i, (_, row) in enumerate(df_norm[df['is_top']].iterrows()):
            ax.plot(cols_to_plot, row[cols_to_plot], color='red', alpha=0.7, linewidth=2)

        # Set the axes
        ax.set_xticks(range(len(cols_to_plot)))
        ax.set_xticklabels(cols_to_plot, rotation=45)
        ax.set_title(f"Parallel Coordinates Plot (Top {top_n} trials highlighted)")

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, trials: List[Dict[str, Any]],
                               target_metric: str = 'score',
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a correlation heatmap of hyperparameters and the target metric.

        Args:
            trials: List of trial dictionaries
            target_metric: The target metric
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not trials:
            logger.warning("No trials provided for correlation heatmap")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
            return fig

        # Convert trials to DataFrame
        df_list = []
        for trial in trials:
            row = trial.get('parameters', {}).copy()
            row[target_metric] = trial.get(target_metric, trial.get('score', 0))
            df_list.append(row)

        df = pd.DataFrame(df_list)

        if df.empty or len(df.columns) <= 1:
            logger.warning("Insufficient data for correlation heatmap")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            return fig

        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            logger.warning("Insufficient numeric data for correlation heatmap")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient numeric data", ha='center', va='center')
            return fig

        # Calculate correlation matrix
        corr = df[numeric_cols].corr()

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        ax.set_title("Parameter Correlation Heatmap")

        plt.tight_layout()
        return fig
