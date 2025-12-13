from __future__ import annotations

import re
from typing import Callable


class KeybindingManager:
    """Manages keyboard shortcuts for Aquarium IDE"""
    
    def __init__(self):
        self.bindings: dict[str, str] = {}
        self.handlers: dict[str, Callable] = {}
    
    def register_keybinding(self, action: str, shortcut: str):
        """Register a keyboard shortcut for an action"""
        self.bindings[action] = shortcut
    
    def register_handler(self, action: str, handler: Callable):
        """Register a handler function for an action"""
        self.handlers[action] = handler
    
    def get_keybinding(self, action: str) -> str | None:
        """Get the keyboard shortcut for an action"""
        return self.bindings.get(action)
    
    def parse_shortcut(self, shortcut: str) -> dict[str, bool | str]:
        """Parse a keyboard shortcut string into components"""
        parts = shortcut.split('+')
        
        modifiers = {
            'ctrl': False,
            'shift': False,
            'alt': False,
            'meta': False
        }
        
        key = ''
        
        for part in parts:
            part_lower = part.lower().strip()
            if part_lower in modifiers:
                modifiers[part_lower] = True
            else:
                key = part.strip()
        
        return {
            'ctrl': modifiers['ctrl'],
            'shift': modifiers['shift'],
            'alt': modifiers['alt'],
            'meta': modifiers['meta'],
            'key': key
        }
    
    def format_shortcut(self, parsed: dict[str, bool | str]) -> str:
        """Format parsed shortcut back to string"""
        parts = []
        if parsed.get('ctrl'):
            parts.append('Ctrl')
        if parsed.get('shift'):
            parts.append('Shift')
        if parsed.get('alt'):
            parts.append('Alt')
        if parsed.get('meta'):
            parts.append('Meta')
        if parsed.get('key'):
            parts.append(str(parsed['key']))
        
        return '+'.join(parts)
    
    def validate_shortcut(self, shortcut: str) -> bool:
        """Validate that a shortcut string is well-formed"""
        pattern = r'^(Ctrl\+|Shift\+|Alt\+|Meta\+)*[A-Za-z0-9`~!@#$%^&*()_\-+=\[\]{};:\'",.<>/?\\|]$'
        return bool(re.match(pattern, shortcut))
    
    def handle_keypress(self, event: dict) -> bool:
        """Handle a keypress event and execute the appropriate action"""
        pressed_shortcut = self._event_to_shortcut(event)
        
        for action, shortcut in self.bindings.items():
            if shortcut == pressed_shortcut and action in self.handlers:
                self.handlers[action]()
                return True
        
        return False
    
    def _event_to_shortcut(self, event: dict) -> str:
        """Convert a keypress event to a shortcut string"""
        parts = []
        if event.get('ctrlKey'):
            parts.append('Ctrl')
        if event.get('shiftKey'):
            parts.append('Shift')
        if event.get('altKey'):
            parts.append('Alt')
        if event.get('metaKey'):
            parts.append('Meta')
        
        key = event.get('key', '')
        if key:
            parts.append(key)
        
        return '+'.join(parts)
    
    def get_all_bindings(self) -> dict[str, str]:
        """Get all registered keybindings"""
        return self.bindings.copy()
    
    def update_bindings(self, bindings: dict[str, str]):
        """Update multiple keybindings at once"""
        self.bindings.update(bindings)
    
    def clear_bindings(self):
        """Clear all keybindings"""
        self.bindings.clear()
