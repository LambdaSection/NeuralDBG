from __future__ import annotations

import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc

from neural.aquarium.src.components.settings import (
    ConfigManager,
    ExtensionManager,
    KeybindingManager,
    SettingsPanel,
    ThemeManager,
)


class AquariumIDE:
    """Aquarium IDE - Neural DSL integrated development environment"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.extension_manager = ExtensionManager(self.config_manager.config_dir)
        self.keybinding_manager = KeybindingManager()
        self.theme_manager = ThemeManager()
        
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True
        )
        
        self.settings_panel = SettingsPanel(self.config_manager)
        
        self._initialize_keybindings()
        self._setup_layout()
        self._register_callbacks()
    
    def _initialize_keybindings(self):
        """Initialize keybindings from configuration"""
        keybindings = self.config_manager.get('keybindings')
        if keybindings:
            self.keybinding_manager.update_bindings(keybindings)
    
    def _setup_layout(self):
        """Setup the main IDE layout"""
        config = self.config_manager.get_all()
        editor_config = config.get('editor', {})
        theme_colors = self.theme_manager.get_theme(
            editor_config.get('theme', 'dark'),
            editor_config.get('custom_theme')
        )
        
        self.app.layout = html.Div([
            # Top menu bar
            dbc.Navbar([
                dbc.Container([
                    dbc.NavbarBrand("Aquarium IDE", className="ms-2"),
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("File", href="#")),
                        dbc.NavItem(dbc.NavLink("Edit", href="#")),
                        dbc.NavItem(dbc.NavLink("View", href="#")),
                        dbc.NavItem(dbc.NavLink("Run", href="#")),
                        dbc.NavItem(dbc.NavLink("Debug", href="#")),
                        dbc.NavItem(dbc.NavLink("Extensions", href="#")),
                        dbc.NavItem(dbc.Button(
                            html.I(className="fas fa-cog"),
                            id="settings-open-btn",
                            color="link",
                            className="text-white"
                        ))
                    ], className="ms-auto", navbar=True)
                ], fluid=True)
            ], color="dark", dark=True, className="mb-0"),
            
            # Main content area
            dbc.Container(fluid=True, children=[
                dbc.Row([
                    # Left sidebar
                    dbc.Col([
                        html.Div([
                            html.H5("Explorer", className="mt-3 mb-3"),
                            html.Div(id="file-explorer", children=[
                                html.P("No folder opened", className="text-muted")
                            ])
                        ], style={
                            'height': '100%',
                            'padding': '10px',
                            'backgroundColor': '#2a2a2a'
                        })
                    ], width=2, style={
                        'height': 'calc(100vh - 56px)',
                        'overflowY': 'auto',
                        'display': (
                            'block'
                            if config.get('ui', {}).get('show_activity_bar', True)
                            else 'none'
                        )
                    }),
                    
                    # Editor area
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(label="model.neural", children=[
                                html.Div([
                                    html.Div(id="editor", children=[
                                        dcc.Textarea(
                                            id="code-editor",
                                            placeholder=(
                                                "Start coding your Neural DSL "
                                                "model here..."
                                            ),
                                            style={
                                                'width': '100%',
                                                'height': 'calc(100vh - 200px)',
                                                'fontFamily': editor_config.get(
                                                    'font_family',
                                                    'Consolas, Monaco, monospace'
                                                ),
                                                'fontSize': (
                                                    f"{editor_config.get('font_size', 14)}px"
                                                ),
                                                **self.theme_manager.apply_theme_to_editor(
                                                    theme_colors
                                                )
                                            }
                                        )
                                    ])
                                ])
                            ])
                        ], id="editor-tabs")
                    ], width=8),
                    
                    # Right panel
                    dbc.Col([
                        html.Div([
                            html.H5("Output", className="mt-3 mb-3"),
                            html.Div(id="output-panel", children=[
                                html.Pre("Ready", style={
                                    'backgroundColor': '#1e1e1e',
                                    'color': '#d4d4d4',
                                    'padding': '10px',
                                    'borderRadius': '5px',
                                    'minHeight': '200px'
                                })
                            ])
                        ], style={
                            'height': '100%',
                            'padding': '10px',
                            'backgroundColor': '#2a2a2a'
                        })
                    ], width=2, style={'height': 'calc(100vh - 56px)', 'overflowY': 'auto'})
                ], className="g-0")
            ], className="p-0"),
            
            # Settings panel
            self.settings_panel.create_layout(),
            
            # Status bar
            html.Div([
                dbc.Container(fluid=True, children=[
                    dbc.Row([
                        dbc.Col([
                            html.Span(
                                (
                                    f"Python: "
                                    f"{config.get('python', {}).get('default_interpreter', 'python')}"
                                ),
                                className="me-3"
                            ),
                            html.Span(
                                (
                                    f"Backend: "
                                    f"{config.get('backend', {}).get('default', 'tensorflow')}"
                                ),
                                className="me-3"
                            ),
                            html.Span(f"Theme: {editor_config.get('theme', 'dark').title()}")
                        ])
                    ])
                ])
            ], style={
                'backgroundColor': '#007acc',
                'color': 'white',
                'padding': '5px 10px',
                'position': 'fixed',
                'bottom': 0,
                'width': '100%',
                'zIndex': 1000,
                'display': 'block' if config.get('ui', {}).get('show_status_bar', True) else 'none'
            })
        ])
    
    def _register_callbacks(self):
        """Register all Dash callbacks"""
        self.settings_panel.register_callbacks(self.app)
        
        @self.app.callback(
            Output("code-editor", "style"),
            Input("settings-store", "data")
        )
        def update_editor_style(settings):
            if not settings:
                return {}
            
            editor_config = settings.get('editor', {})
            theme_colors = self.theme_manager.get_theme(
                editor_config.get('theme', 'dark'),
                editor_config.get('custom_theme')
            )
            
            return {
                'width': '100%',
                'height': 'calc(100vh - 200px)',
                'fontFamily': editor_config.get('font_family', 'Consolas, Monaco, monospace'),
                'fontSize': f"{editor_config.get('font_size', 14)}px",
                **self.theme_manager.apply_theme_to_editor(theme_colors)
            }
    
    def run(self, debug=True, port=8052):
        """Run the Aquarium IDE application"""
        print(f"Starting Aquarium IDE on http://localhost:{port}")
        print("Press Ctrl+C to quit")
        self.app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    ide = AquariumIDE()
    ide.run()
