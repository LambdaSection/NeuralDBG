from __future__ import annotations

from typing import Any


class ThemeManager:
    """Manages editor themes for Aquarium IDE"""
    
    LIGHT_THEME = {
        'background': '#ffffff',
        'foreground': '#000000',
        'selection': '#add6ff',
        'comment': '#008000',
        'keyword': '#0000ff',
        'string': '#a31515',
        'number': '#098658',
        'function': '#795e26',
        'operator': '#000000',
        'cursor': '#000000'
    }
    
    DARK_THEME = {
        'background': '#1e1e1e',
        'foreground': '#d4d4d4',
        'selection': '#264f78',
        'comment': '#6a9955',
        'keyword': '#569cd6',
        'string': '#ce9178',
        'number': '#b5cea8',
        'function': '#dcdcaa',
        'operator': '#d4d4d4',
        'cursor': '#ffffff'
    }
    
    @staticmethod
    def get_theme(theme_name: str, custom_theme: dict[str, str] | None = None) -> dict[str, str]:
        """Get theme colors by name"""
        if theme_name == 'light':
            return ThemeManager.LIGHT_THEME.copy()
        elif theme_name == 'dark':
            return ThemeManager.DARK_THEME.copy()
        elif theme_name == 'custom' and custom_theme:
            return custom_theme.copy()
        return ThemeManager.DARK_THEME.copy()
    
    @staticmethod
    def apply_theme_to_editor(theme_colors: dict[str, str]) -> dict[str, Any]:
        """Convert theme colors to editor CSS properties"""
        return {
            'backgroundColor': theme_colors.get('background', '#1e1e1e'),
            'color': theme_colors.get('foreground', '#d4d4d4'),
            'caretColor': theme_colors.get('cursor', '#ffffff')
        }
    
    @staticmethod
    def generate_theme_css(theme_colors: dict[str, str]) -> str:
        """Generate CSS string for theme"""
        css = f"""
        .editor {{
            background-color: {theme_colors.get('background', '#1e1e1e')};
            color: {theme_colors.get('foreground', '#d4d4d4')};
        }}
        
        .editor .selection {{
            background-color: {theme_colors.get('selection', '#264f78')};
        }}
        
        .editor .comment {{
            color: {theme_colors.get('comment', '#6a9955')};
            font-style: italic;
        }}
        
        .editor .keyword {{
            color: {theme_colors.get('keyword', '#569cd6')};
            font-weight: bold;
        }}
        
        .editor .string {{
            color: {theme_colors.get('string', '#ce9178')};
        }}
        
        .editor .number {{
            color: {theme_colors.get('number', '#b5cea8')};
        }}
        
        .editor .function {{
            color: {theme_colors.get('function', '#dcdcaa')};
        }}
        
        .editor .operator {{
            color: {theme_colors.get('operator', '#d4d4d4')};
        }}
        
        .editor .cursor {{
            border-left: 2px solid {theme_colors.get('cursor', '#ffffff')};
        }}
        """
        return css
