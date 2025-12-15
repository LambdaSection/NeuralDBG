"""
Standalone script to launch the experiment comparison UI.

Usage:
    python neural/tracking/comparison_ui.py
    python neural/tracking/comparison_ui.py --port 8052 --base-dir ./my_experiments
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from neural.tracking import ExperimentManager
from neural.tracking.comparison_component import ComparisonComponent


class ExperimentComparisonUI:
    """Web UI for comparing experiments."""
    
    def __init__(self, manager: ExperimentManager, port: int = 8052):
        """
        Initialize the comparison UI.
        
        Args:
            manager: ExperimentManager instance
            port: Port to run the UI on
        """
        self.manager = manager
        self.port = port
        
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True,
        )
        
        self.app.title = "Neural Experiment Comparison"
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the UI layout."""
        self.app.layout = dbc.Container(
            [
                dcc.Store(id="selected-experiments", data=[]),
                html.H1(
                    [
                        html.I(className="fas fa-chart-line me-2"),
                        "Neural Experiment Comparison",
                    ],
                    className="text-primary mb-3",
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Select Experiments"),
                                dcc.Dropdown(
                                    id="experiment-selector",
                                    multi=True,
                                    placeholder="Select experiments to compare...",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    [html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                                    id="refresh-button",
                                    color="primary",
                                    className="mt-4",
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.Hr(),
                html.Div(id="comparison-content"),
            ],
            fluid=True,
            style={"padding": "20px"},
        )
    
    def _setup_callbacks(self):
        """Setup UI callbacks."""
        
        @self.app.callback(
            Output("experiment-selector", "options"),
            [Input("refresh-button", "n_clicks")],
        )
        def update_experiment_options(n_clicks):
            """Update the list of available experiments."""
            experiments = self.manager.list_experiments()
            return [
                {
                    "label": f"{exp['experiment_name']} ({exp['experiment_id'][:8]}) - {exp['status']}",
                    "value": exp["experiment_id"],
                }
                for exp in experiments
            ]
        
        @self.app.callback(
            Output("comparison-content", "children"),
            [Input("experiment-selector", "value")],
        )
        def update_comparison(experiment_ids):
            """Update the comparison view."""
            if not experiment_ids:
                return dbc.Alert(
                    [
                        html.I(className="fas fa-info-circle me-2"),
                        "Select experiments from the dropdown above to compare.",
                    ],
                    color="info",
                )
            
            experiments = [
                self.manager.get_experiment(exp_id)
                for exp_id in experiment_ids
                if self.manager.get_experiment(exp_id)
            ]
            
            if not experiments:
                return dbc.Alert("No valid experiments selected.", color="warning")
            
            component = ComparisonComponent(experiments)
            return component.render()
    
    def run(self, debug: bool = False, host: str = "127.0.0.1"):
        """Run the UI server."""
        print(f"\nStarting Neural Experiment Comparison UI...")
        print(f"URL: http://{host}:{self.port}")
        print()
        
        experiments = self.manager.list_experiments()
        print(f"Found {len(experiments)} experiments\n")
        
        self.app.run_server(debug=debug, host=host, port=self.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Neural Experiment Comparison UI"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="neural_experiments",
        help="Base directory containing experiments (default: neural_experiments)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8052,
        help="Port to run the UI on (default: 8052)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the UI on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    args = parser.parse_args()

    print(f"Starting Neural Experiment Comparison UI...")
    print(f"Base directory: {args.base_dir}")
    print(f"URL: http://{args.host}:{args.port}")
    print()

    manager = ExperimentManager(base_dir=args.base_dir)
    
    experiments = manager.list_experiments()
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments[:5]:
        print(f"  - {exp['experiment_name']} ({exp['experiment_id']}) - {exp['status']}")
    if len(experiments) > 5:
        print(f"  ... and {len(experiments) - 5} more")
    print()

    ui = ExperimentComparisonUI(manager=manager, port=args.port)
    ui.run(debug=args.debug, host=args.host)


if __name__ == "__main__":
    main()
