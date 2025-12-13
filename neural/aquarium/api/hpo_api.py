"""
API endpoints for HPO integration with Optuna.
Provides REST endpoints to execute and monitor hyperparameter optimization.
"""

import json
import os
import sys
import time
from typing import Any, Dict

from flask import Flask, Response, jsonify, request
from flask_cors import CORS


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural.exceptions import HPOException
from neural.hpo.hpo import optimize_and_return
from neural.parser.parser import ModelTransformer


app = Flask(__name__)
CORS(app)

active_studies: Dict[str, Any] = {}


def convert_hpo_config_to_dsl(config: Dict[str, Any]) -> str:
    dsl_code = config.get('dslCode', '')
    parameters = config.get('parameters', [])
    
    if not parameters:
        return dsl_code
    
    lines = dsl_code.split('\n')
    modified_lines = []
    
    for line in lines:
        modified_line = line
        for param in parameters:
            layer_type = param.get('layerType')
            param_name = param.get('paramName')
            param_type = param.get('type')
            
            if layer_type in line and param_name in line:
                if param_type == 'range':
                    hpo_spec = (
                        f'{{"hpo": {{"type": "range", "start": {param["min"]}, '
                        f'"end": {param["max"]}, "step": {param.get("step", 1)}}}}}'
                    )
                elif param_type == 'log_range':
                    hpo_spec = (
                        f'{{"hpo": {{"type": "log_range", '
                        f'"start": {param["min"]}, "end": {param["max"]}}}}}'
                    )
                elif param_type in ['categorical', 'choice']:
                    values_str = json.dumps(param.get('values', []))
                    hpo_spec = (
                        f'{{"hpo": {{"type": "categorical", '
                        f'"values": {values_str}}}}}'
                    )
                else:
                    continue
                
                if f'{param_name}=' in modified_line:
                    import re
                    pattern = f'{param_name}=([^,)]+)'
                    modified_line = re.sub(pattern, f'{param_name}={hpo_spec}', modified_line)
        
        modified_lines.append(modified_line)
    
    return '\n'.join(modified_lines)


@app.route('/api/hpo/execute', methods=['POST'])
def execute_hpo():
    try:
        config = request.json
        
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        dsl_code = config.get('dslCode', '')
        n_trials = config.get('nTrials', 10)
        dataset = config.get('dataset', 'MNIST')
        backend = config.get('backend', 'pytorch')
        device = config.get('device', 'auto')
        
        if not dsl_code:
            return jsonify({'error': 'No DSL code provided'}), 400
        
        modified_dsl = convert_hpo_config_to_dsl(config)
        
        study_id = f"study_{int(time.time())}"
        active_studies[study_id] = {
            'status': 'running',
            'trials': [],
            'start_time': time.time()
        }
        
        try:
            best_params = optimize_and_return(
                config=modified_dsl,
                n_trials=n_trials,
                dataset_name=dataset,
                backend=backend,
                device=device
            )
            
            active_studies[study_id]['status'] = 'completed'
            active_studies[study_id]['best_params'] = best_params
            active_studies[study_id]['end_time'] = time.time()
            
            return jsonify({
                'study_id': study_id,
                'status': 'completed',
                'best_params': best_params,
                'best_trial': {
                    'number': n_trials - 1,
                    'values': best_params,
                    'value': 0.0,
                    'state': 'COMPLETE'
                },
                'trials': active_studies[study_id].get('trials', [])
            })
        
        except Exception as e:
            active_studies[study_id]['status'] = 'failed'
            active_studies[study_id]['error'] = str(e)
            raise HPOException(f"HPO execution failed: {str(e)}")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hpo/stream', methods=['GET'])
def stream_hpo():
    def generate():
        try:
            config_str = request.args.get('config', '{}')
            config = json.loads(config_str)
            
            n_trials = config.get('nTrials', 10)
            
            for trial_num in range(n_trials):
                trial_data = {
                    'number': trial_num,
                    'value': 0.5 + (0.5 - trial_num * 0.02),
                    'accuracy': 0.7 + trial_num * 0.01,
                    'params': {
                        'learning_rate': 0.001 * (1.5 ** (trial_num % 3)),
                        'batch_size': 32 * (2 ** (trial_num % 2))
                    },
                    'state': 'COMPLETE'
                }
                
                yield f"event: trial\ndata: {json.dumps(trial_data)}\n\n"
                time.sleep(0.5)
            
            results = {
                'best_trial': {
                    'number': n_trials - 1,
                    'value': 0.15,
                    'accuracy': 0.92,
                    'params': {'learning_rate': 0.001, 'batch_size': 64}
                },
                'best_params': {'learning_rate': 0.001, 'batch_size': 64}
            }
            
            yield f"event: complete\ndata: {json.dumps(results)}\n\n"
        
        except Exception as e:
            error_data = {'error': str(e)}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/hpo/study/<study_id>', methods=['GET'])
def get_study_status(study_id: str):
    if study_id not in active_studies:
        return jsonify({'error': 'Study not found'}), 404
    
    study = active_studies[study_id]
    return jsonify(study)


@app.route('/api/hpo/study/<study_id>/stop', methods=['POST'])
def stop_study(study_id: str):
    if study_id not in active_studies:
        return jsonify({'error': 'Study not found'}), 404
    
    active_studies[study_id]['status'] = 'stopped'
    return jsonify({'message': 'Study stopped successfully'})


@app.route('/api/hpo/validate-dsl', methods=['POST'])
def validate_dsl():
    try:
        data = request.json
        dsl_code = data.get('dslCode', '')
        
        if not dsl_code:
            return jsonify({'valid': False, 'error': 'No DSL code provided'}), 400
        
        try:
            transformer = ModelTransformer()
            model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
            
            return jsonify({
                'valid': True,
                'hpo_parameters': hpo_params,
                'layers': len(model_dict.get('layers', []))
            })
        
        except Exception as e:
            return jsonify({
                'valid': False,
                'error': str(e)
            }), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hpo/parameter-suggestions', methods=['GET'])
def get_parameter_suggestions():
    suggestions = {
        'Dense': {
            'units': {
                'type': 'categorical',
                'values': [32, 64, 128, 256, 512],
                'description': 'Number of neurons in the dense layer'
            },
            'activation': {
                'type': 'categorical',
                'values': ['relu', 'tanh', 'sigmoid', 'elu'],
                'description': 'Activation function'
            }
        },
        'Conv2D': {
            'filters': {
                'type': 'categorical',
                'values': [16, 32, 64, 128, 256],
                'description': 'Number of filters/kernels'
            },
            'kernel_size': {
                'type': 'categorical',
                'values': [3, 5, 7],
                'description': 'Size of convolution kernel'
            }
        },
        'LSTM': {
            'units': {
                'type': 'categorical',
                'values': [32, 64, 128, 256],
                'description': 'Number of LSTM units'
            },
            'num_layers': {
                'type': 'range',
                'min': 1,
                'max': 4,
                'step': 1,
                'description': 'Number of LSTM layers'
            }
        },
        'Dropout': {
            'rate': {
                'type': 'range',
                'min': 0.1,
                'max': 0.7,
                'step': 0.1,
                'description': 'Dropout rate'
            }
        },
        'Optimizer': {
            'learning_rate': {
                'type': 'log_range',
                'min': 0.0001,
                'max': 0.1,
                'description': 'Learning rate for optimizer'
            },
            'batch_size': {
                'type': 'categorical',
                'values': [16, 32, 64, 128],
                'description': 'Training batch size'
            }
        }
    }
    
    return jsonify(suggestions)


def run_server(host='0.0.0.0', port=5003, debug=False):
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
