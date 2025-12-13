"""
Demo script showing how to use the Aquarium IDE settings components
"""
from neural.aquarium.src.components.settings import (
    ConfigManager,
    ExtensionManager,
    KeybindingManager,
    ThemeManager,
)


def demo_config_manager():
    """Demonstrate ConfigManager usage"""
    print("=" * 60)
    print("ConfigManager Demo")
    print("=" * 60)
    
    config = ConfigManager()
    
    # Get configuration values
    print("\n1. Getting configuration values:")
    theme = config.get('editor', 'theme')
    print(f"   Current editor theme: {theme}")
    
    font_size = config.get('editor', 'font_size')
    print(f"   Current font size: {font_size}")
    
    # Get entire section
    keybindings = config.get('keybindings')
    print("\n2. Keybindings section:")
    for action, shortcut in list(keybindings.items())[:3]:
        print(f"   {action}: {shortcut}")
    
    # Update configuration
    print("\n3. Updating configuration:")
    original_theme = config.get('editor', 'theme')
    config.set('editor', 'theme', 'light')
    print(f"   Changed theme from '{original_theme}' to 'light'")
    
    # Update multiple values
    config.update_section('editor', {
        'font_size': 16,
        'line_numbers': True
    })
    print("   Updated font size to 16 and enabled line numbers")
    
    # Restore original
    config.set('editor', 'theme', original_theme)
    print(f"   Restored theme to '{original_theme}'")
    
    print("\n   Configuration file: " + str(config.config_file))


def demo_theme_manager():
    """Demonstrate ThemeManager usage"""
    print("\n" + "=" * 60)
    print("ThemeManager Demo")
    print("=" * 60)
    
    # Get predefined themes
    print("\n1. Predefined themes:")
    dark_theme = ThemeManager.get_theme('dark')
    print(f"   Dark theme background: {dark_theme['background']}")
    print(f"   Dark theme foreground: {dark_theme['foreground']}")
    
    light_theme = ThemeManager.get_theme('light')
    print(f"   Light theme background: {light_theme['background']}")
    print(f"   Light theme foreground: {light_theme['foreground']}")
    
    # Create custom theme
    print("\n2. Custom theme:")
    custom_theme = {
        'background': '#2b2b2b',
        'foreground': '#f8f8f2',
        'selection': '#49483e',
        'comment': '#75715e',
        'keyword': '#f92672',
        'string': '#e6db74',
        'number': '#ae81ff',
        'function': '#a6e22e',
        'operator': '#f8f8f2',
        'cursor': '#f8f8f0'
    }
    theme_colors = ThemeManager.get_theme('custom', custom_theme)
    print(f"   Custom theme keyword color: {theme_colors['keyword']}")
    
    # Apply theme to editor
    print("\n3. Applying theme to editor:")
    editor_style = ThemeManager.apply_theme_to_editor(dark_theme)
    print(f"   Editor style: {editor_style}")
    
    # Generate CSS
    print("\n4. Generate CSS (first 200 chars):")
    css = ThemeManager.generate_theme_css(dark_theme)
    print(f"   {css[:200]}...")


def demo_keybinding_manager():
    """Demonstrate KeybindingManager usage"""
    print("\n" + "=" * 60)
    print("KeybindingManager Demo")
    print("=" * 60)
    
    kb_manager = KeybindingManager()
    
    # Register keybindings
    print("\n1. Registering keybindings:")
    kb_manager.register_keybinding('save', 'Ctrl+S')
    kb_manager.register_keybinding('open', 'Ctrl+O')
    kb_manager.register_keybinding('save_as', 'Ctrl+Shift+S')
    print("   Registered: save, open, save_as")
    
    # Get keybinding
    print("\n2. Getting keybindings:")
    save_kb = kb_manager.get_keybinding('save')
    print(f"   'save' keybinding: {save_kb}")
    
    # Parse shortcut
    print("\n3. Parsing shortcuts:")
    parsed = kb_manager.parse_shortcut('Ctrl+Shift+S')
    print(f"   Parsed 'Ctrl+Shift+S': {parsed}")
    
    formatted = kb_manager.format_shortcut(parsed)
    print(f"   Formatted back: {formatted}")
    
    # Validate shortcuts
    print("\n4. Validating shortcuts:")
    valid_shortcuts = ['Ctrl+S', 'Ctrl+Shift+P', 'Alt+F4', 'F5']
    invalid_shortcuts = ['Ctrl++', 'InvalidKey', '']
    
    for shortcut in valid_shortcuts:
        is_valid = kb_manager.validate_shortcut(shortcut)
        print(f"   '{shortcut}': {is_valid}")
    
    print("\n   Invalid examples:")
    for shortcut in invalid_shortcuts:
        is_valid = kb_manager.validate_shortcut(shortcut)
        print(f"   '{shortcut}': {is_valid}")


def demo_extension_manager():
    """Demonstrate ExtensionManager usage"""
    print("\n" + "=" * 60)
    print("ExtensionManager Demo")
    print("=" * 60)
    
    
    config = ConfigManager()
    ext_manager = ExtensionManager(config.config_dir)
    
    print("\n1. Extension manager initialized")
    print(f"   Extensions directory: {ext_manager.extensions_dir}")
    
    # Get all extensions
    print("\n2. Installed extensions:")
    all_exts = ext_manager.get_all_extensions()
    if all_exts:
        for ext in all_exts:
            status = "Enabled" if ext.enabled else "Disabled"
            print(f"   {ext.name} v{ext.version} [{status}]")
    else:
        print("   No extensions installed")
    
    # Get enabled/disabled extensions
    enabled = ext_manager.get_enabled_extensions()
    disabled = ext_manager.get_disabled_extensions()
    print("\n3. Extension counts:")
    print(f"   Enabled: {len(enabled)}")
    print(f"   Disabled: {len(disabled)}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("Aquarium IDE Settings Demo")
    print("=" * 60)
    
    demo_config_manager()
    demo_theme_manager()
    demo_keybinding_manager()
    demo_extension_manager()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\nConfiguration is stored in: ~/.aquarium/config.json")
    print("You can manually edit this file or use the IDE settings panel.")


if __name__ == "__main__":
    main()
