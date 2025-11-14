"""
Automated Test Runner and Reporter

Runs tests and generates reports.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class TestAutomation:
    """Automate test running and reporting."""
    
    def __init__(self, tests_dir: str = "tests"):
        """Initialize test automation."""
        self.tests_dir = Path(tests_dir)
        self.results = {}
    
    def run_tests(self, verbose: bool = True, coverage: bool = False) -> Dict:
        """
        Run test suite.
        
        Args:
            verbose: Show verbose output
            coverage: Generate coverage report
            
        Returns:
            Test results dictionary
        """
        print("Running tests...")
        
        cmd = [sys.executable, "-m", "pytest", str(self.tests_dir)]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=neural", "--cov-report=html", "--cov-report=term"])
        
        # Add JSON report
        cmd.extend(["--json-report", "--json-report-file=test_results.json"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            # Parse JSON report if available
            json_results = {}
            if os.path.exists("test_results.json"):
                with open("test_results.json", "r") as f:
                    json_results = json.load(f)
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "json": json_results
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Tests timed out after 10 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_report(self, results: Dict, output_file: str = "test_report.md"):
        """Generate test report."""
        report = f"""# Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""
        
        if results.get("success"):
            report += "✅ **All tests passed!**\n\n"
        else:
            report += "❌ **Some tests failed**\n\n"
        
        if "json" in results and results["json"]:
            json_data = results["json"]
            summary = json_data.get("summary", {})
            
            report += f"""
- **Total Tests**: {summary.get('total', 'N/A')}
- **Passed**: {summary.get('passed', 'N/A')} ✅
- **Failed**: {summary.get('failed', 'N/A')} ❌
- **Skipped**: {summary.get('skipped', 'N/A')} ⏭️
- **Duration**: {summary.get('duration', 'N/A')}s

"""
        
        if not results.get("success") and results.get("stdout"):
            report += "## Test Output\n\n```\n"
            report += results["stdout"][-2000:]  # Last 2000 chars
            report += "\n```\n"
        
        if results.get("stderr"):
            report += "\n## Errors\n\n```\n"
            report += results["stderr"][-1000:]  # Last 1000 chars
            report += "\n```\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✓ Report saved to: {output_file}")
        return report
    
    def run_and_report(self, coverage: bool = False):
        """Run tests and generate report."""
        results = self.run_tests(coverage=coverage)
        self.generate_report(results)
        return results


if __name__ == "__main__":
    automation = TestAutomation()
    automation.run_and_report(coverage=True)

