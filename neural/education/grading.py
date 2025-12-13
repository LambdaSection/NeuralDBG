"""
Automated grading system for Neural DSL assignments.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from lark import Lark
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False


@dataclass
class GradingCriteria:
    """Criteria for grading an assignment."""
    syntax_valid: int = 10
    architecture_valid: int = 20
    layer_count_min: Optional[int] = None
    layer_count_max: Optional[int] = None
    performance_threshold: Optional[float] = None
    code_quality: int = 20
    documentation: int = 10
    creativity: int = 10
    custom_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    def total_points(self) -> int:
        """Calculate total available points."""
        return (
            self.syntax_valid
            + self.architecture_valid
            + self.code_quality
            + self.documentation
            + self.creativity
            + sum(test.get("points", 0) for test in self.custom_tests)
        )


@dataclass
class GradingResult:
    """Result of grading an assignment."""
    total_score: float
    max_score: float
    percentage: float
    breakdown: Dict[str, float]
    feedback: List[str]
    errors: List[str]
    warnings: List[str]
    passed_tests: int
    total_tests: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "breakdown": self.breakdown,
            "feedback": self.feedback,
            "errors": self.errors,
            "warnings": self.warnings,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
        }


class AutoGrader:
    """Automated grading engine for Neural DSL code."""
    
    def __init__(self, criteria: Optional[GradingCriteria] = None):
        self.criteria = criteria or GradingCriteria()
    
    def grade(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> GradingResult:
        """Grade a student's submission."""
        metadata = metadata or {}
        breakdown = {}
        feedback = []
        errors = []
        warnings = []
        passed_tests = 0
        total_tests = 0
        
        syntax_score, syntax_feedback = self._check_syntax(code)
        breakdown["syntax"] = syntax_score
        feedback.extend(syntax_feedback)
        
        if syntax_score < self.criteria.syntax_valid:
            errors.append("Code contains syntax errors")
        
        arch_score, arch_feedback = self._check_architecture(code)
        breakdown["architecture"] = arch_score
        feedback.extend(arch_feedback)
        
        quality_score, quality_feedback = self._check_code_quality(code)
        breakdown["code_quality"] = quality_score
        feedback.extend(quality_feedback)
        
        doc_score, doc_feedback = self._check_documentation(code)
        breakdown["documentation"] = doc_score
        feedback.extend(doc_feedback)
        
        creativity_score, creativity_feedback = self._check_creativity(code, metadata)
        breakdown["creativity"] = creativity_score
        feedback.extend(creativity_feedback)
        
        for test in self.criteria.custom_tests:
            total_tests += 1
            passed, test_feedback = self._run_custom_test(code, test, metadata)
            if passed:
                passed_tests += 1
                test_points = test.get("points", 0)
                breakdown[f"test_{test.get('name', 'custom')}"] = test_points
            feedback.extend(test_feedback)
        
        total_score = sum(breakdown.values())
        max_score = self.criteria.total_points()
        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0
        
        if percentage >= 90:
            feedback.append("Excellent work! 🎉")
        elif percentage >= 80:
            feedback.append("Great job! Keep it up! 👏")
        elif percentage >= 70:
            feedback.append("Good effort. Review the feedback for improvement areas.")
        else:
            feedback.append("Needs improvement. Please review the requirements carefully.")
        
        return GradingResult(
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            breakdown=breakdown,
            feedback=feedback,
            errors=errors,
            warnings=warnings,
            passed_tests=passed_tests,
            total_tests=total_tests,
        )
    
    def _check_syntax(self, code: str) -> Tuple[float, List[str]]:
        """Check if code has valid Neural DSL syntax."""
        feedback = []
        score = 0.0
        
        if not LARK_AVAILABLE:
            feedback.append("Lark parser not available, skipping syntax check")
            return self.criteria.syntax_valid, feedback
        
        try:
            if "network" in code and "{" in code and "}" in code:
                score = self.criteria.syntax_valid
                feedback.append("✓ Valid DSL syntax")
            else:
                feedback.append("✗ Missing network definition")
        except Exception as e:
            feedback.append(f"✗ Syntax error: {str(e)}")
        
        return score, feedback
    
    def _check_architecture(self, code: str) -> Tuple[float, List[str]]:
        """Check if architecture meets requirements."""
        feedback = []
        score = 0.0
        points_per_check = self.criteria.architecture_valid / 3
        
        if re.search(r'network\s+\w+\s*{', code):
            score += points_per_check
            feedback.append("✓ Network definition found")
        else:
            feedback.append("✗ No network definition found")
            return score, feedback
        
        layer_pattern = r'layer\s+\w+\s*:\s*\w+'
        layers = re.findall(layer_pattern, code)
        num_layers = len(layers)
        
        if num_layers > 0:
            score += points_per_check
            feedback.append(f"✓ Found {num_layers} layers")
            
            if self.criteria.layer_count_min and num_layers < self.criteria.layer_count_min:
                feedback.append(f"⚠ Fewer than {self.criteria.layer_count_min} layers required")
            elif self.criteria.layer_count_max and num_layers > self.criteria.layer_count_max:
                feedback.append(f"⚠ More than {self.criteria.layer_count_max} layers specified")
            else:
                score += points_per_check
                feedback.append("✓ Layer count meets requirements")
        else:
            feedback.append("✗ No layers defined")
        
        if re.search(r'flow\s*:', code):
            feedback.append("✓ Data flow defined")
        else:
            feedback.append("⚠ No data flow defined")
        
        return score, feedback
    
    def _check_code_quality(self, code: str) -> Tuple[float, List[str]]:
        """Check code quality metrics."""
        feedback = []
        score = 0.0
        points_per_check = self.criteria.code_quality / 4
        
        if len(code.split('\n')) <= 200:
            score += points_per_check
            feedback.append("✓ Code is concise")
        else:
            feedback.append("⚠ Code is quite long, consider simplification")
        
        if re.search(r'activation\s*=', code):
            score += points_per_check
            feedback.append("✓ Activation functions specified")
        
        consistent_indent = True
        lines = code.split('\n')
        for line in lines:
            if line.strip() and not (line.startswith(' ' * 4) or line.startswith('\t') or not line.startswith(' ')):
                consistent_indent = True
                break
        
        if consistent_indent:
            score += points_per_check
            feedback.append("✓ Consistent indentation")
        
        if re.search(r'(Conv2D|Dense|LSTM|GRU|Attention)', code):
            score += points_per_check
            feedback.append("✓ Uses appropriate layer types")
        
        return score, feedback
    
    def _check_documentation(self, code: str) -> Tuple[float, List[str]]:
        """Check documentation quality."""
        feedback = []
        score = 0.0
        points_per_check = self.criteria.documentation / 2
        
        comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]
        
        if len(comment_lines) >= 3:
            score += points_per_check
            feedback.append(f"✓ Good documentation ({len(comment_lines)} comments)")
        elif len(comment_lines) > 0:
            score += points_per_check / 2
            feedback.append("⚠ Could use more comments")
        else:
            feedback.append("✗ No comments found")
        
        if re.search(r'#.*(?:layer|network|input|output)', code, re.IGNORECASE):
            score += points_per_check
            feedback.append("✓ Comments explain architecture")
        else:
            feedback.append("⚠ Add comments explaining your architecture choices")
        
        return score, feedback
    
    def _check_creativity(
        self,
        code: str,
        metadata: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Check for creative or advanced techniques."""
        feedback = []
        score = 0.0
        points_per_feature = self.criteria.creativity / 5
        
        advanced_layers = [
            'Attention', 'MultiHeadAttention', 'LayerNorm', 
            'BatchNorm', 'Dropout', 'ResidualConnection'
        ]
        
        found_advanced = [layer for layer in advanced_layers if layer in code]
        if found_advanced:
            score += points_per_feature * min(len(found_advanced), 3)
            feedback.append(f"✓ Uses advanced techniques: {', '.join(found_advanced)}")
        
        if re.search(r'skip\s+connection|residual', code, re.IGNORECASE):
            score += points_per_feature
            feedback.append("✓ Implements skip connections")
        
        if metadata.get("custom_approach"):
            score += points_per_feature
            feedback.append("✓ Creative approach to the problem")
        
        return score, feedback
    
    def _run_custom_test(
        self,
        code: str,
        test: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Run a custom test case."""
        feedback = []
        test_name = test.get("name", "custom test")
        test_type = test.get("type", "regex")
        
        if test_type == "regex":
            pattern = test.get("pattern", "")
            if re.search(pattern, code):
                feedback.append(f"✓ Passed {test_name}")
                return True, feedback
            else:
                feedback.append(f"✗ Failed {test_name}")
                return False, feedback
        
        elif test_type == "layer_count":
            min_count = test.get("min", 0)
            max_count = test.get("max", float('inf'))
            layers = re.findall(r'layer\s+\w+\s*:\s*\w+', code)
            count = len(layers)
            
            if min_count <= count <= max_count:
                feedback.append(f"✓ Passed {test_name} (found {count} layers)")
                return True, feedback
            else:
                feedback.append(f"✗ Failed {test_name} (expected {min_count}-{max_count}, got {count})")
                return False, feedback
        
        elif test_type == "contains":
            required = test.get("required", [])
            missing = [item for item in required if item not in code]
            
            if not missing:
                feedback.append(f"✓ Passed {test_name}")
                return True, feedback
            else:
                feedback.append(f"✗ Failed {test_name}: missing {', '.join(missing)}")
                return False, feedback
        
        elif test_type == "metadata":
            key = test.get("key", "")
            expected = test.get("expected")
            actual = metadata.get(key)
            
            if actual == expected:
                feedback.append(f"✓ Passed {test_name}")
                return True, feedback
            else:
                feedback.append(f"✗ Failed {test_name}: expected {expected}, got {actual}")
                return False, feedback
        
        feedback.append(f"⚠ Unknown test type: {test_type}")
        return False, feedback
    
    def generate_feedback_report(self, result: GradingResult) -> str:
        """Generate a formatted feedback report."""
        report = []
        report.append("=" * 60)
        report.append(f"Grade: {result.percentage:.1f}% ({result.total_score}/{result.max_score} points)")
        report.append("=" * 60)
        report.append("")
        
        report.append("Score Breakdown:")
        for category, score in result.breakdown.items():
            report.append(f"  {category}: {score:.1f} points")
        report.append("")
        
        if result.errors:
            report.append("Errors:")
            for error in result.errors:
                report.append(f"  • {error}")
            report.append("")
        
        if result.warnings:
            report.append("Warnings:")
            for warning in result.warnings:
                report.append(f"  • {warning}")
            report.append("")
        
        report.append("Feedback:")
        for item in result.feedback:
            report.append(f"  {item}")
        report.append("")
        
        if result.total_tests > 0:
            report.append(f"Tests Passed: {result.passed_tests}/{result.total_tests}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
