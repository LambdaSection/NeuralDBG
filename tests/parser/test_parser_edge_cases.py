import os
import sys

from lark.exceptions import VisitError
import pytest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.parser.parser import (
    DSLValidationError,
    ModelTransformer,
    Severity,
    create_parser,
    custom_error_handler,
    log_by_severity,
    safe_parse,
    split_params,
)


class TestErrorHandlingAndValidation:
    """Test error handling and validation edge cases"""
    
    def test_custom_error_handler_key_error(self):
        """Test custom error handler with KeyError"""
        error = KeyError("test key error")
        with pytest.raises(DSLValidationError) as exc_info:
            custom_error_handler(error)
        assert "Unexpected end of input" in str(exc_info.value)
    
    def test_custom_error_handler_unknown_error(self):
        """Test custom error handler with unknown error type"""
        error = RuntimeError("Unknown runtime error")
        with pytest.raises(DSLValidationError) as exc_info:
            custom_error_handler(error)
        assert "Unknown runtime error" in str(exc_info.value)
    
    def test_safe_parse_with_syntax_error(self):
        """Test safe_parse with syntax errors"""
        parser = create_parser('network')
        invalid_code = "network Test { input: (10,) layers:"
        with pytest.raises(DSLValidationError) as exc_info:
            safe_parse(parser, invalid_code)
        assert "Unexpected" in str(exc_info.value)
    
    def test_safe_parse_with_invalid_characters(self):
        """Test safe_parse with invalid characters"""
        parser = create_parser('network')
        invalid_code = "network Test { input: (10,) layers: Dense(10) & }"
        with pytest.raises(DSLValidationError):
            safe_parse(parser, invalid_code)
    
    def test_dsl_validation_error_with_line_column(self):
        """Test DSLValidationError with line and column information"""
        error = DSLValidationError("Test error", Severity.ERROR, line=5, column=10)
        assert "line 5" in str(error)
        assert "column 10" in str(error)
    
    def test_dsl_validation_error_without_line_column(self):
        """Test DSLValidationError without line and column information"""
        error = DSLValidationError("Test error", Severity.WARNING)
        assert "WARNING" in str(error)
        assert "line" not in str(error).lower()


class TestParserEdgeCases:
    """Test parser edge cases and boundary conditions"""
    
    def test_empty_input(self):
        """Test parsing empty input"""
        parser = create_parser('network')
        with pytest.raises(DSLValidationError):
            parser.parse("")
    
    def test_whitespace_only(self):
        """Test parsing whitespace-only input"""
        parser = create_parser('network')
        with pytest.raises(DSLValidationError):
            parser.parse("   \n\t  ")
    
    def test_nested_braces_mismatched(self):
        """Test mismatched nested braces"""
        parser = create_parser('network')
        code = "network Test { input: (10,) layers: Dense(10) { "
        with pytest.raises(DSLValidationError):
            safe_parse(parser, code)
    
    def test_multiple_hpo_expressions_in_single_param(self):
        """Test multiple HPO expressions in a single parameter"""
        create_parser('network')
        transformer = ModelTransformer()
        code = """
        network Test {
            input: (10,)
            layers: Dense(HPO(choice(64, 128)), activation=HPO(choice("relu", "tanh")))
        }
        """
        result = transformer.parse_network(code)
        assert 'hpo' in result['layers'][0]['params']['units']
        assert 'hpo' in result['layers'][0]['params']['activation']
    
    def test_layer_with_empty_sublayers(self):
        """Test layer with empty sublayers block"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "ResidualConnection() { }"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['sublayers'] == []
    
    def test_very_large_numeric_values(self):
        """Test very large numeric values"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dense(999999999)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['units'] == 999999999
    
    def test_negative_numeric_values(self):
        """Test negative numeric values where invalid"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dense(-10)"
        with pytest.raises(VisitError):
            tree = parser.parse(code)
            transformer.transform(tree)
    
    def test_scientific_notation_in_params(self):
        """Test scientific notation in parameters"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dropout(1e-3)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert abs(result['params']['rate'] - 0.001) < 1e-9
    
    def test_split_params_with_nested_parens(self):
        """Test split_params with nested parentheses"""
        result = split_params("64, (3, 3), activation='relu'")
        assert len(result) == 3
        assert result[1] == "(3, 3)"
    
    def test_split_params_empty_string(self):
        """Test split_params with empty string"""
        result = split_params("")
        assert result == ['']
    
    def test_split_params_no_commas(self):
        """Test split_params with no commas"""
        result = split_params("single_param")
        assert result == ['single_param']


class TestTransformerMethods:
    """Test ModelTransformer methods and edge cases"""
    
    def test_raise_validation_error_with_item(self):
        """Test raise_validation_error with parse tree item"""
        transformer = ModelTransformer()
        from lark import Token
        token = Token('NAME', 'test')
        with pytest.raises(DSLValidationError) as exc_info:
            transformer.raise_validation_error("Test error", token, Severity.ERROR)
        assert "Test error" in str(exc_info.value)
    
    def test_raise_validation_error_without_item(self):
        """Test raise_validation_error without parse tree item"""
        transformer = ModelTransformer()
        with pytest.raises(DSLValidationError) as exc_info:
            transformer.raise_validation_error("Test error", None, Severity.ERROR)
        assert "Test error" in str(exc_info.value)
    
    def test_raise_validation_error_warning(self):
        """Test raise_validation_error with warning severity"""
        transformer = ModelTransformer()
        result = transformer.raise_validation_error("Warning message", None, Severity.WARNING)
        assert result['warning'] == "Warning message"
    
    def test_validate_input_dimensions_negative(self):
        """Test input dimension validation with negative values"""
        transformer = ModelTransformer()
        with pytest.raises(DSLValidationError):
            transformer._validate_input_dimensions((10, -5, 20))
    
    def test_validate_input_dimensions_zero(self):
        """Test input dimension validation with zero values"""
        transformer = ModelTransformer()
        with pytest.raises(DSLValidationError):
            transformer._validate_input_dimensions((10, 0, 20))
    
    def test_validate_optimizer_invalid(self):
        """Test optimizer validation with invalid optimizer"""
        transformer = ModelTransformer()
        with pytest.raises(DSLValidationError):
            transformer._validate_optimizer("InvalidOptimizer")
    
    def test_validate_optimizer_case_insensitive(self):
        """Test optimizer validation is case-insensitive"""
        transformer = ModelTransformer()
        # Should not raise error
        transformer._validate_optimizer("ADAM")
        transformer._validate_optimizer("Adam")
        transformer._validate_optimizer("adam")
    
    def test_validate_loss_function_invalid(self):
        """Test loss function validation with invalid loss"""
        transformer = ModelTransformer()
        with pytest.raises(DSLValidationError):
            transformer._validate_loss_function("invalid_loss")
    
    def test_validate_loss_function_valid(self):
        """Test loss function validation with valid losses"""
        transformer = ModelTransformer()
        # Should not raise errors
        transformer._validate_loss_function("mse")
        transformer._validate_loss_function("categorical_crossentropy")
        transformer._validate_loss_function("binary_cross_entropy")
    
    def test_extract_layer_def_none(self):
        """Test _extract_layer_def with None"""
        transformer = ModelTransformer()
        result = transformer._extract_layer_def(None)
        assert result is None
    
    def test_extract_layer_def_invalid(self):
        """Test _extract_layer_def with invalid layer definition"""
        transformer = ModelTransformer()
        from lark import Token
        token = Token('STRING', '"not a dict"')
        with pytest.raises(DSLValidationError):
            transformer._extract_layer_def(token)


class TestLayerParsing:
    """Test layer parsing edge cases"""
    
    def test_dense_with_float_units_conversion(self):
        """Test Dense layer with float units (should convert to int)"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dense(64.0)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['units'] == 64
        assert isinstance(result['params']['units'], int)
    
    def test_dense_with_multiple_activations(self):
        """Test Dense layer doesn't accept multiple activations"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dense(64, 'relu', 'tanh')"
        with pytest.raises(VisitError):
            tree = parser.parse(code)
            transformer.transform(tree)
    
    def test_conv2d_missing_kernel_size(self):
        """Test Conv2D without kernel_size parameter"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Conv2D(filters=32)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        # Should parse but may have issues during shape propagation
        assert result['params']['filters'] == 32
    
    def test_dropout_rate_exactly_zero(self):
        """Test Dropout with rate exactly 0"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dropout(0.0)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['rate'] == 0.0
    
    def test_dropout_rate_exactly_one(self):
        """Test Dropout with rate exactly 1"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dropout(1.0)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['rate'] == 1.0
    
    def test_dropout_negative_rate(self):
        """Test Dropout with negative rate"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dropout(-0.5)"
        with pytest.raises(VisitError):
            tree = parser.parse(code)
            transformer.transform(tree)
    
    def test_output_layer_without_activation(self):
        """Test Output layer without activation parameter"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Output(10)"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['units'] == 10
        assert 'activation' not in result['params']
    
    def test_batchnorm_no_params(self):
        """Test BatchNormalization with no parameters"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "BatchNormalization()"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['type'] == 'BatchNormalization'
    
    def test_flatten_with_params(self):
        """Test Flatten with parameters (should be ignored)"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Flatten()"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['type'] == 'Flatten'


class TestMacroHandling:
    """Test macro definition and reference edge cases"""
    
    def test_macro_with_no_layers(self):
        """Test macro definition with no layers"""
        parser = create_parser('define')
        transformer = ModelTransformer()
        code = "define EmptyMacro { }"
        with pytest.raises(VisitError):
            tree = parser.parse(code)
            transformer.transform(tree)
    
    def test_undefined_macro_reference(self):
        """Test reference to undefined macro"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "UndefinedMacro()"
        with pytest.raises(VisitError):
            tree = parser.parse(code)
            transformer.transform(tree)
    
    def test_macro_definition_storage(self):
        """Test that macro definition is stored correctly"""
        parser = create_parser('define')
        transformer = ModelTransformer()
        code = "define TestMacro { Dense(64) }"
        tree = parser.parse(code)
        transformer.transform(tree)
        assert 'TestMacro' in transformer.macros
        assert transformer.macros['TestMacro']['macro']['type'] == 'TestMacro'


class TestNetworkParsing:
    """Test network-level parsing edge cases"""
    
    def test_network_with_multiple_inputs(self):
        """Test network with multiple input specifications"""
        transformer = ModelTransformer()
        code = """
        network MultiInput {
            input: (10,), (20,)
            layers: Dense(5)
        }
        """
        with pytest.raises(DSLValidationError):
            transformer.parse_network(code)
    
    def test_network_with_named_inputs(self):
        """Test network with named inputs"""
        transformer = ModelTransformer()
        code = """
        network NamedInput {
            input: { input1: (10,), input2: (20,) }
            layers: Dense(5)
        }
        """
        result = transformer.parse_network(code)
        assert 'input1' in result['input']
        assert 'input2' in result['input']
    
    def test_network_missing_optimizer(self):
        """Test network without optimizer (should use default)"""
        transformer = ModelTransformer()
        code = """
        network NoOptimizer {
            input: (10,)
            layers: Dense(5)
        }
        """
        result = transformer.parse_network(code)
        assert result['optimizer'] is not None
    
    def test_network_missing_loss(self):
        """Test network without loss (should use default)"""
        transformer = ModelTransformer()
        code = """
        network NoLoss {
            input: (10,)
            layers: Dense(5)
        }
        """
        result = transformer.parse_network(code)
        assert result['loss'] is not None
    
    def test_network_with_training_config_zero_epochs(self):
        """Test network with zero epochs"""
        transformer = ModelTransformer()
        code = """
        network ZeroEpochs {
            input: (10,)
            layers: Dense(5)
            train { epochs: 0 }
        }
        """
        with pytest.raises(DSLValidationError):
            transformer.parse_network(code)
    
    def test_network_with_training_config_negative_batch_size(self):
        """Test network with negative batch size"""
        transformer = ModelTransformer()
        code = """
        network NegativeBatch {
            input: (10,)
            layers: Dense(5)
            train { batch_size: -32 }
        }
        """
        with pytest.raises(DSLValidationError):
            transformer.parse_network(code)
    
    def test_network_with_execution_config(self):
        """Test network with execution configuration"""
        transformer = ModelTransformer()
        code = """
        network WithExecution {
            input: (10,)
            layers: Dense(5)
            execute { device: "cpu" }
        }
        """
        result = transformer.parse_network(code)
        assert 'execution_config' in result


class TestHPOExpressions:
    """Test hyperparameter optimization expressions"""
    
    def test_hpo_choice_with_single_value(self):
        """Test HPO choice with only one value"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dense(HPO(choice(64)))"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert 'hpo' in result['params']['units']
        assert result['params']['units']['hpo']['values'] == [64]
    
    def test_hpo_range_with_step(self):
        """Test HPO range with explicit step"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dropout(HPO(range(0.1, 0.9, step=0.1)))"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert 'hpo' in result['params']['rate']
        assert result['params']['rate']['hpo']['step'] == 0.1
    
    def test_hpo_log_range(self):
        """Test HPO log range"""
        transformer = ModelTransformer()
        code = """
        network HPOLog {
            input: (10,)
            layers: Dense(64)
            optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.1)))
        }
        """
        result = transformer.parse_network(code)
        assert 'hpo' in result['optimizer']['params']['learning_rate']
        assert result['optimizer']['params']['learning_rate']['hpo']['type'] == 'log_range'
    
    def test_hpo_choice_mixed_types(self):
        """Test HPO choice with mixed types (should work)"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = "Dense(HPO(choice(32, 64, 128)))"
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert 'hpo' in result['params']['units']


class TestDeviceSpecification:
    """Test device specification edge cases"""
    
    def test_valid_device_cpu(self):
        """Test valid CPU device specification"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = 'Dense(64) @ "cpu"'
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['device'] == 'cpu'
    
    def test_valid_device_cuda_with_index(self):
        """Test valid CUDA device with index"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = 'Dense(64) @ "cuda:0"'
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['device'] == 'cuda:0'
    
    def test_valid_device_tpu(self):
        """Test valid TPU device specification"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = 'Dense(64) @ "tpu:0"'
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['params']['device'] == 'tpu:0'
    
    def test_invalid_device_specification(self):
        """Test invalid device specification"""
        parser = create_parser('layer')
        transformer = ModelTransformer()
        code = 'Dense(64) @ "npu"'
        with pytest.raises(VisitError):
            tree = parser.parse(code)
            transformer.transform(tree)


class TestSeverityLevels:
    """Test severity level handling"""
    
    def test_log_by_severity_debug(self):
        """Test logging with DEBUG severity"""
        # Should not raise error
        log_by_severity(Severity.DEBUG, "Debug message")
    
    def test_log_by_severity_info(self):
        """Test logging with INFO severity"""
        # Should not raise error
        log_by_severity(Severity.INFO, "Info message")
    
    def test_log_by_severity_warning(self):
        """Test logging with WARNING severity"""
        # Should not raise error
        log_by_severity(Severity.WARNING, "Warning message")
    
    def test_log_by_severity_error(self):
        """Test logging with ERROR severity"""
        # Should not raise error (just logs)
        log_by_severity(Severity.ERROR, "Error message")
    
    def test_log_by_severity_critical(self):
        """Test logging with CRITICAL severity"""
        # Should not raise error (just logs)
        log_by_severity(Severity.CRITICAL, "Critical message")


class TestResearchParsing:
    """Test research file parsing edge cases"""
    
    def test_research_with_no_name(self):
        """Test research without name"""
        parser = create_parser('research')
        transformer = ModelTransformer()
        code = """
        research {
            metrics {
                accuracy: 0.95
            }
        }
        """
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['type'] == 'Research'
        assert result['name'] is None
    
    def test_research_with_only_references(self):
        """Test research with only references"""
        parser = create_parser('research')
        transformer = ModelTransformer()
        code = """
        research OnlyRefs {
            references {
                paper: "Paper 1"
            }
        }
        """
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert 'references' in result.get('params', {})
    
    def test_research_empty_metrics(self):
        """Test research with empty metrics block"""
        parser = create_parser('research')
        transformer = ModelTransformer()
        code = """
        research EmptyMetrics {
            metrics { }
        }
        """
        tree = parser.parse(code)
        result = transformer.transform(tree)
        assert result['type'] == 'Research'


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
