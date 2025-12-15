#!/usr/bin/env python3
"""Cleanup script to remove deprecated integration files."""

import os
from pathlib import Path

files_to_remove = [
    'neural/integrations/runai.py',
    'neural/integrations/sagemaker.py',
    'neural/integrations/vertex_ai.py',
]

for filepath in files_to_remove:
    path = Path(filepath)
    if path.exists():
        path.unlink()
        print(f"Removed: {filepath}")
    else:
        print(f"Not found: {filepath}")

print("\nCleanup complete!")
