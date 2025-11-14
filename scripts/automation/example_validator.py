"""
Automated Example Validator

Validates all examples in the examples/ directory.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class ExampleValidator:
    """Validate Neural DSL examples."""
    
    def __init__(self, examples_dir: str = "examples"):
        """Initialize validator."""
        self.examples_dir = Path(examples_dir)
        self.results = []
    
    def find_examples(self) -> List[Path]:
        """Find all .neural files in examples directory."""
        if not self.examples_dir.exists():
            return []
        
        examples = list(self.examples_dir.glob("*.neural"))
        return examples
    
    def validate_example(self, example_path: Path) -> Tuple[bool, str]:
        """
        Validate a single example.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Try to parse the example
            result = subprocess.run(
                [sys.executable, "-c", 
                 f"from neural.parser.parser import ModelTransformer; "
                 f"from neural.parser.parser import create_parser; "
                 f"parser = create_parser('network'); "
                 f"transformer = ModelTransformer(); "
                 f"with open('{example_path}', 'r') as f: "
                 f"  tree = parser.parse(f.read()); "
                 f"  model = transformer.transform(tree); "
                 f"  print('OK')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Validation timed out"
        except Exception as e:
            return False, str(e)
    
    def validate_all(self) -> Dict[str, any]:
        """Validate all examples."""
        examples = self.find_examples()
        
        if not examples:
            return {"total": 0, "valid": 0, "invalid": 0, "results": []}
        
        print(f"Found {len(examples)} examples to validate...")
        
        results = []
        valid_count = 0
        invalid_count = 0
        
        for example in examples:
            print(f"Validating {example.name}...", end=" ")
            is_valid, error = self.validate_example(example)
            
            if is_valid:
                print("✓")
                valid_count += 1
            else:
                print("✗")
                print(f"  Error: {error[:100]}")
                invalid_count += 1
            
            results.append({
                "file": str(example),
                "valid": is_valid,
                "error": error
            })
        
        return {
            "total": len(examples),
            "valid": valid_count,
            "invalid": invalid_count,
            "results": results
        }
    
    def generate_report(self, output_file: str = "examples_validation_report.md"):
        """Generate validation report."""
        results = self.validate_all()
        
        report = f"""# Example Validation Report

Generated: {Path(__file__).stat().st_mtime}

## Summary

- **Total Examples**: {results['total']}
- **Valid**: {results['valid']} ✓
- **Invalid**: {results['invalid']} ✗

## Results

"""
        
        for result in results['results']:
            status = "✓" if result['valid'] else "✗"
            report += f"### {status} {Path(result['file']).name}\n\n"
            
            if not result['valid']:
                report += f"**Error**:\n```\n{result['error']}\n```\n\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {output_file}")
        return report


if __name__ == "__main__":
    validator = ExampleValidator()
    validator.generate_report()

