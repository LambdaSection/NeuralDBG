#!/usr/bin/env python3
"""
Batch script to replace print() statements with proper logging throughout the neural/ directory.
Excludes CLI output files, examples, and test files where print() is appropriate.
"""

import os
import re
from pathlib import Path

# Directories and files to exclude from replacement
EXCLUDE_PATTERNS = [
    r'\\cli\\',  # CLI files use print for user output
    r'\\examples\\',  # Example files use print for demonstration
    r'\\test',  # Test files
    r'_test\.py$',  # Test files
    r'\\__pycache__',  # Cache
    r'\\welcome_message\.py$',  # Welcome message uses print for display
    r'\\cli_aesthetics\.py$',  # CLI aesthetics use print for display
    r'\\benchmarks\\',  # Benchmarks use print for output
]

def should_process_file(filepath):
    """Check if file should be processed."""
    filepath_str = str(filepath)
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, filepath_str):
            return False
    return True

def add_logger_import(content, filepath):
    """Add logger import if not present."""
    # Check if logging is already imported
    if re.search(r'^import logging', content, re.MULTILINE):
        # Check if logger is defined
        if not re.search(r'^logger = logging\.getLogger\(__name__\)', content, re.MULTILINE):
            # Add logger definition after logging import
            content = re.sub(
                r'(^import logging.*$)',
                r'\1\n\nlogger = logging.getLogger(__name__)',
                content,
                count=1,
                flags=re.MULTILINE
            )
    else:
        # Add both logging import and logger definition at the top
        # Find the first non-docstring, non-comment line
        lines = content.split('\n')
        insert_pos = 0
        in_docstring = False
        docstring_char = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Handle docstrings
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_char = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(docstring_char) >= 2:
                    # Single-line docstring
                    continue
                else:
                    in_docstring = True
                continue
            elif in_docstring and docstring_char in stripped:
                in_docstring = False
                continue
            elif in_docstring:
                continue
            
            # Skip comments and empty lines
            if not stripped or stripped.startswith('#'):
                continue
            
            # Found first real code line
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_pos = i
                break
            else:
                insert_pos = i
                break
        
        # Insert logging import
        if insert_pos == 0:
            content = 'import logging\n\nlogger = logging.getLogger(__name__)\n\n' + content
        else:
            lines.insert(insert_pos, 'import logging\n\nlogger = logging.getLogger(__name__)\n')
            content = '\n'.join(lines)
    
    return content

def replace_print_statements(content):
    """Replace print() statements with appropriate logging calls."""
    # Pattern to match print statements
    print_patterns = [
        # print(f"DEBUG: ...") -> logger.debug(...)
        (r'print\(f["\']DEBUG:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.debug(f"\1", \2)'),
        (r'print\(f["\']DEBUG:?\s*([^"\']*)["\'](?:, |%\s*)(.+?)\)', r'logger.debug(f"\1", \2)'),
        (r'print\(f["\']DEBUG:?\s*([^"\']*)["\'\)]', r'logger.debug(f"\1")'),
        (r'print\(["\']DEBUG:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.debug("\1", \2)'),
        (r'print\(["\']DEBUG:?\s*([^"\']*)["\'](?:%|\s*,\s*)(.+?)\)', r'logger.debug("\1", \2)'),
        (r'print\(["\']DEBUG:?\s*([^"\']*)["\'\)]', r'logger.debug("\1")'),
        
        # print(f"WARNING: ...") -> logger.warning(...)
        (r'print\(f["\']WARNING:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.warning(f"\1", \2)'),
        (r'print\(f["\']WARNING:?\s*([^"\']*)["\'](?:, |%\s*)(.+?)\)', r'logger.warning(f"\1", \2)'),
        (r'print\(f["\']WARNING:?\s*([^"\']*)["\'\)]', r'logger.warning(f"\1")'),
        (r'print\(["\']WARNING:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.warning("\1", \2)'),
        (r'print\(["\']WARNING:?\s*([^"\']*)["\'](?:%|\s*,\s*)(.+?)\)', r'logger.warning("\1", \2)'),
        (r'print\(["\']WARNING:?\s*([^"\']*)["\'\)]', r'logger.warning("\1")'),
        
        # print(f"ERROR: ...") -> logger.error(...)
        (r'print\(f["\']ERROR:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.error(f"\1", \2)'),
        (r'print\(f["\']ERROR:?\s*([^"\']*)["\'](?:, |%\s*)(.+?)\)', r'logger.error(f"\1", \2)'),
        (r'print\(f["\']ERROR:?\s*([^"\']*)["\'\)]', r'logger.error(f"\1")'),
        (r'print\(["\']ERROR:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.error("\1", \2)'),
        (r'print\(["\']ERROR:?\s*([^"\']*)["\'](?:%|\s*,\s*)(.+?)\)', r'logger.error("\1", \2)'),
        (r'print\(["\']ERROR:?\s*([^"\']*)["\'\)]', r'logger.error("\1")'),
        
        # print(f"INFO: ...") -> logger.info(...)
        (r'print\(f["\']INFO:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.info(f"\1", \2)'),
        (r'print\(f["\']INFO:?\s*([^"\']*)["\'](?:, |%\s*)(.+?)\)', r'logger.info(f"\1", \2)'),
        (r'print\(f["\']INFO:?\s*([^"\']*)["\'\)]', r'logger.info(f"\1")'),
        (r'print\(["\']INFO:?\s*([^"\']*)["\']\.format\((.*?)\)\)', r'logger.info("\1", \2)'),
        (r'print\(["\']INFO:?\s*([^"\']*)["\'](?:%|\s*,\s*)(.+?)\)', r'logger.info("\1", \2)'),
        (r'print\(["\']INFO:?\s*([^"\']*)["\'\)]', r'logger.info("\1")'),
        
        # print(f"TRACE: ...") -> logger.debug(...)
        (r'print\(f["\']TRACE:?\s*([^"\']*)["\'\)]', r'logger.debug(f"\1")'),
        (r'print\(["\']TRACE:?\s*([^"\']*)["\'\)]', r'logger.debug("\1")'),
        
        # Generic print(f"...") -> logger.info(f"...")
        (r'print\(f["\']([^"\']*)["\']\)', r'logger.info(f"\1")'),
        
        # Generic print("...") -> logger.info("...")
        (r'print\(["\']([^"\']*)["\']\)', r'logger.info("\1")'),
        
        # print(...) with variables -> logger.info(...)
        (r'print\(([^)]+)\)', r'logger.info(\1)'),
    ]
    
    for pattern, replacement in print_patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def process_file(filepath):
    """Process a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Check if file has print statements
        if 'print(' not in original_content:
            return False
        
        # Add logger import if needed
        content = add_logger_import(original_content, filepath)
        
        # Replace print statements
        content = replace_print_statements(content)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main execution function."""
    neural_dir = Path('neural')
    processed = 0
    skipped = 0
    
    # Find all Python files
    python_files = list(neural_dir.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files in neural/")
    
    for filepath in python_files:
        if should_process_file(filepath):
            if process_file(filepath):
                processed += 1
                print(f"Processed: {filepath}")
        else:
            skipped += 1
    
    print(f"\nComplete!")
    print(f"Processed: {processed} files")
    print(f"Skipped: {skipped} files")

if __name__ == '__main__':
    main()
