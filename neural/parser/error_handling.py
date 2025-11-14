"""
Enhanced error handling for the Neural parser.
Provides detailed, context-aware error messages and recovery strategies.

This module implements a comprehensive error handling system that:
- Provides precise error location (line/column)
- Shows surrounding code context
- Suggests fixes for common mistakes
- Categorizes errors by severity
- Offers actionable solutions
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from lark import UnexpectedToken, UnexpectedCharacters
import difflib
import re

@dataclass
class ParserError:
    """Structured representation of a parsing error with full context."""
    message: str
    line: int
    column: int
    context: str
    suggestion: Optional[str] = None
    error_type: Optional[str] = None  # 'syntax', 'validation', 'shape', 'semantic'
    severity: str = "ERROR"  # 'ERROR', 'WARNING', 'INFO'
    fix_hint: Optional[str] = None  # Specific fix suggestion

class NeuralParserError(Exception):
    """Custom exception class for Neural parser errors."""
    def __init__(self, error: ParserError):
        self.error = error
        super().__init__(str(error))

class ErrorHandler:
    """Handles and formats parser errors with helpful context and suggestions.
    
    This class provides intelligent error handling that:
    1. Detects common typos and suggests corrections
    2. Provides code context around errors
    3. Offers specific fix suggestions
    4. Categorizes errors for better user experience
    """
    
    # Common typos and their corrections
    COMMON_MISTAKES = {
        "Dense": ["dense", "Dence", "DNse", "Dens"],
        "Conv2D": ["conv2d", "Conv2d", "conv2D", "Con2D", "Conv2"],
        "Conv1D": ["conv1d", "Conv1d", "conv1D", "Con1D"],
        "Conv3D": ["conv3d", "Conv3d", "conv3D", "Con3D"],
        "Input": ["input", "input_layer", "Iput", "Inpt"],
        "Output": ["output", "output_layer", "Ouput", "Outpt"],
        "MaxPooling2D": ["maxpooling2d", "MaxPooling2d", "max_pooling2d", "MaxPool2D"],
        "Flatten": ["flatten", "Flaten", "Flatn"],
        "Dropout": ["dropout", "DropOut", "drop_out"],
        "activation": ["activate", "activations", "activ", "actvation"],
        "filters": ["filter", "filers", "filtes"],
        "kernel_size": ["kernel_size", "kernal_size", "kernelSize", "kernel"],
        "units": ["unit", "unites", "unts"],
        "pool_size": ["poolsize", "poolSize", "pool_size"],
    }
    
    # Valid layer types for suggestions
    VALID_LAYERS = [
        "Dense", "Conv2D", "Conv1D", "Conv3D", "MaxPooling2D", "MaxPooling1D", "MaxPooling3D",
        "Flatten", "Dropout", "Input", "Output", "LSTM", "GRU", "BatchNormalization",
        "GlobalAveragePooling2D", "ResidualConnection", "Concatenate", "Add"
    ]
    
    # Common parameter names
    VALID_PARAMS = [
        "filters", "kernel_size", "units", "activation", "pool_size", "strides",
        "padding", "rate", "dropout", "units", "input_shape", "output_shape"
    ]
    
    @staticmethod
    def get_line_context(code: str, line_no: int, context_lines: int = 3, column: int = None) -> str:
        """Get the surrounding lines of code for context with error indicator.
        
        Args:
            code: The source code
            line_no: Line number (0-indexed)
            context_lines: Number of lines before/after to show
            column: Column number for error indicator (optional)
        
        Returns:
            Formatted context string with line numbers and error indicator
        """
        lines = code.splitlines()
        start = max(0, line_no - context_lines)
        end = min(len(lines), line_no + context_lines + 1)
        
        context = []
        for i in range(start, end):
            line_content = lines[i] if i < len(lines) else ""
            prefix = ">>>" if i == line_no else "   "
            line_num = f"{i+1:4d}"
            context.append(f"{prefix} {line_num} | {line_content}")
            
            # Add error indicator arrow if column is specified
            if i == line_no and column is not None:
                # Create arrow pointing to error column
                arrow = " " * (column + 11) + "^" * max(1, len(line_content[column:column+10]) if column < len(line_content) else 1)
                context.append(f"     {arrow}")
        
        return "\n".join(context)
    
    @staticmethod
    def suggest_correction(token: str, context: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Suggest corrections for common mistakes.
        
        Args:
            token: The incorrect token
            context: Optional surrounding context for better suggestions
        
        Returns:
            Tuple of (suggested_correction, fix_hint)
        """
        token_lower = token.lower()
        
        # Check common mistakes first
        for correct, mistakes in ErrorHandler.COMMON_MISTAKES.items():
            if token_lower in [m.lower() for m in mistakes]:
                fix_hint = f"Replace '{token}' with '{correct}'"
                return correct, fix_hint
        
        # Check if it's a layer name typo
        layer_matches = difflib.get_close_matches(token, ErrorHandler.VALID_LAYERS, n=1, cutoff=0.6)
        if layer_matches:
            fix_hint = f"Did you mean the layer type '{layer_matches[0]}'?"
            return layer_matches[0], fix_hint
        
        # Check if it's a parameter name typo
        param_matches = difflib.get_close_matches(token, ErrorHandler.VALID_PARAMS, n=1, cutoff=0.7)
        if param_matches:
            fix_hint = f"Did you mean the parameter '{param_matches[0]}'?"
            return param_matches[0], fix_hint
        
        # General fuzzy matching
        all_valid = ErrorHandler.VALID_LAYERS + ErrorHandler.VALID_PARAMS
        matches = difflib.get_close_matches(token, all_valid, n=1, cutoff=0.5)
        if matches:
            fix_hint = f"Did you mean '{matches[0]}'?"
            return matches[0], fix_hint
        
        return None, None
    
    @staticmethod
    def detect_common_issues(code: str, line_no: int, column: int) -> Optional[str]:
        """Detect common issues and provide specific fix hints.
        
        Args:
            code: The source code
            line_no: Line number (0-indexed)
            column: Column number
        
        Returns:
            Fix hint string or None
        """
        lines = code.splitlines()
        if line_no >= len(lines):
            return None
        
        line = lines[line_no]
        
        # Check for missing colon
        if "layers:" not in line and "input:" not in line and "loss:" not in line:
            if re.search(r'\w+\s*$', line) and line_no < len(lines) - 1:
                next_line = lines[line_no + 1] if line_no + 1 < len(lines) else ""
                if next_line.strip().startswith(("Conv", "Dense", "Flatten")):
                    return "Missing colon (:) after network property. Add ':' after the property name."
        
        # Check for missing parentheses
        if re.search(r'\b(Conv2D|Dense|Flatten|MaxPooling2D)\s*$', line):
            return "Missing opening parenthesis '(' after layer name. Add '(' to start parameters."
        
        # Check for unclosed parentheses
        open_parens = line.count('(')
        close_parens = line.count(')')
        if open_parens > close_parens:
            return f"Missing {open_parens - close_parens} closing parenthesis(es) ')'"
        
        # Check for missing quotes in strings
        if re.search(r'activation\s*=\s*[^"\'][^,)]+[^"\']', line):
            return "String values (like activation functions) should be in quotes. Use: activation=\"relu\""
        
        return None
    
    @classmethod
    def handle_unexpected_token(cls, error: UnexpectedToken, code: str) -> ParserError:
        """Handle unexpected token errors with enhanced context and suggestions.
        
        Args:
            error: The UnexpectedToken exception from Lark
            code: The source code being parsed
        
        Returns:
            ParserError with full context and suggestions
        """
        line_no = error.line - 1  # Convert to 0-indexed
        context = cls.get_line_context(code, line_no, column=error.column - 1)
        
        token_str = str(error.token)
        suggestion, fix_hint = cls.suggest_correction(token_str, context)
        
        # Detect common issues
        common_issue = cls.detect_common_issues(code, line_no, error.column - 1)
        
        # Build comprehensive error message
        msg = f"Unexpected token '{token_str}' at line {error.line}, column {error.column}"
        
        if error.expected:
            expected_str = ", ".join(sorted(set(str(e) for e in error.expected))[:5])
            msg += f". Expected one of: {expected_str}"
        
        if suggestion:
            msg += f"\n💡 Suggestion: Did you mean '{suggestion}'?"
        
        if fix_hint:
            msg += f"\n🔧 Fix: {fix_hint}"
        
        if common_issue:
            msg += f"\n⚠️  Common Issue: {common_issue}"
            
        return ParserError(
            message=msg,
            line=error.line,
            column=error.column,
            context=context,
            suggestion=suggestion,
            error_type="syntax",
            severity="ERROR",
            fix_hint=fix_hint or common_issue
        )
    
    @classmethod
    def handle_unexpected_char(cls, error: UnexpectedCharacters, code: str) -> ParserError:
        """Handle unexpected character errors with enhanced context.
        
        Args:
            error: The UnexpectedCharacters exception from Lark
            code: The source code being parsed
        
        Returns:
            ParserError with full context and suggestions
        """
        line_no = error.line - 1  # Convert to 0-indexed
        context = cls.get_line_context(code, line_no, column=error.column - 1)
        
        # Detect common issues
        common_issue = cls.detect_common_issues(code, line_no, error.column - 1)
        
        msg = f"Unexpected character '{error.char}' at line {error.line}, column {error.column}"
        
        if hasattr(error, 'allowed') and error.allowed:
            allowed_str = ", ".join(sorted(error.allowed)[:5])
            msg += f". Expected one of: {allowed_str}"
        
        if common_issue:
            msg += f"\n⚠️  Common Issue: {common_issue}"
        
        return ParserError(
            message=msg,
            line=error.line,
            column=error.column,
            context=context,
            error_type="syntax",
            severity="ERROR",
            fix_hint=common_issue
        )
    
    @classmethod
    def handle_shape_error(cls, shape_error: Exception, code: str, line_no: int, layer_name: str = None) -> ParserError:
        """Handle shape propagation errors with enhanced context and suggestions.
        
        Args:
            shape_error: The shape error exception
            code: The source code
            line_no: Line number (0-indexed)
            layer_name: Optional layer name for better context
        
        Returns:
            ParserError with shape-specific suggestions
        """
        context = cls.get_line_context(code, line_no)
        
        error_msg = str(shape_error)
        
        # Add shape-specific fix hints
        fix_hint = None
        if "dimension" in error_msg.lower() or "shape" in error_msg.lower():
            fix_hint = "Check that input/output shapes match between layers. Use 'neural visualize' to see shape flow."
        elif "mismatch" in error_msg.lower():
            fix_hint = "Shape mismatch detected. Verify layer parameters (filters, kernel_size, etc.) are compatible."
        
        if layer_name:
            error_msg = f"Shape error in {layer_name}: {error_msg}"
        
        return ParserError(
            message=error_msg,
            line=line_no + 1,  # Convert to 1-indexed for display
            column=0,
            context=context,
            error_type="shape",
            severity="ERROR",
            fix_hint=fix_hint
        )
    
    @classmethod
    def format_error(cls, error: ParserError) -> str:
        """Format a ParserError into a user-friendly string.
        
        Args:
            error: The ParserError to format
        
        Returns:
            Formatted error message string
        """
        lines = [
            f"\n{'='*70}",
            f"{error.severity}: {error.error_type.upper() if error.error_type else 'PARSER'} ERROR",
            f"{'='*70}",
            f"\n{error.message}",
            f"\n📍 Location: Line {error.line}, Column {error.column}",
            f"\n📄 Context:",
            error.context,
        ]
        
        if error.suggestion:
            lines.append(f"\n💡 Suggestion: {error.suggestion}")
        
        if error.fix_hint:
            lines.append(f"\n🔧 Fix Hint: {error.fix_hint}")
        
        lines.append(f"\n{'='*70}\n")
        
        return "\n".join(lines)