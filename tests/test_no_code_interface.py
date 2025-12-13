"""
Tests for the No-Code Interface
"""

import json
import pytest
from neural.no_code.app import app, config


@pytest.fixture
def client():
    """Create a test client"""
    config.TESTING = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Test index route returns HTML"""
    response = client.get('/')
    assert response.status_code == 200


def test_layers_api(client):
    """Test layers API returns layer categories"""
    response = client.get('/api/layers')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'Convolutional' in data
    assert 'Core' in data
    assert 'Recurrent' in data


def test_templates_api(client):
    """Test templates API returns model templates"""
    response = client.get('/api/templates')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'mnist_cnn' in data
    assert 'cifar10_vgg' in data
    assert 'text_lstm' in data


def test_tutorial_api(client):
    """Test tutorial API returns tutorial steps"""
    response = client.get('/api/tutorial')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    assert 'title' in data[0]
    assert 'content' in data[0]


def test_validate_empty_model(client):
    """Test validation with empty model"""
    response = client.post('/api/validate', json={
        'input_shape': [None, 28, 28, 1],
        'layers': []
    })
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['valid'] == True
    assert len(data['warnings']) > 0


def test_validate_simple_model(client):
    """Test validation with simple model"""
    response = client.post('/api/validate', json={
        'input_shape': [None, 28, 28, 1],
        'layers': [
            {
                'id': 'node-1',
                'type': 'Conv2D',
                'params': {
                    'filters': 32,
                    'kernel_size': [3, 3],
                    'activation': 'relu'
                }
            },
            {
                'id': 'node-2',
                'type': 'Flatten',
                'params': {}
            },
            {
                'id': 'node-3',
                'type': 'Dense',
                'params': {
                    'units': 10,
                    'activation': 'softmax'
                }
            }
        ]
    })
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'valid' in data
    assert 'shapes' in data


def test_generate_code(client):
    """Test code generation"""
    response = client.post('/api/generate-code', json={
        'input_shape': [None, 28, 28, 1],
        'layers': [
            {
                'type': 'Dense',
                'params': {
                    'units': 10,
                    'activation': 'softmax'
                }
            }
        ],
        'optimizer': {
            'type': 'Adam',
            'params': {
                'learning_rate': 0.001
            }
        },
        'loss': 'categorical_crossentropy'
    })
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'dsl' in data
    assert 'tensorflow' in data
    assert 'pytorch' in data
    assert 'network MyModel' in data['dsl']


def test_save_and_load_model(client):
    """Test saving and loading a model"""
    model_data = {
        'name': 'test_model',
        'input_shape': [None, 28, 28, 1],
        'layers': [
            {
                'type': 'Dense',
                'params': {
                    'units': 10,
                    'activation': 'softmax'
                }
            }
        ]
    }
    
    # Save model
    save_response = client.post('/api/save', json=model_data)
    assert save_response.status_code == 200
    
    save_data = json.loads(save_response.data)
    assert save_data['success'] == True
    
    # Load model
    load_response = client.get('/api/load/test_model')
    assert load_response.status_code == 200
    
    load_data = json.loads(load_response.data)
    assert load_data['name'] == 'test_model'


def test_list_models(client):
    """Test listing saved models"""
    response = client.get('/api/models')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert isinstance(data, list)


def test_load_nonexistent_model(client):
    """Test loading a model that doesn't exist"""
    response = client.get('/api/load/nonexistent_model')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data


def test_validate_invalid_shapes(client):
    """Test validation with incompatible shapes"""
    response = client.post('/api/validate', json={
        'input_shape': [None, 28, 28, 1],
        'layers': [
            {
                'id': 'node-1',
                'type': 'Dense',
                'params': {
                    'units': 10
                }
            }
        ]
    })
    assert response.status_code == 200
    
    data = json.loads(response.data)


def test_complex_model_validation(client):
    """Test validation with complex model"""
    response = client.post('/api/validate', json={
        'input_shape': [None, 224, 224, 3],
        'layers': [
            {
                'id': 'node-1',
                'type': 'Conv2D',
                'params': {
                    'filters': 64,
                    'kernel_size': [7, 7],
                    'strides': [2, 2],
                    'padding': 'same'
                }
            },
            {
                'id': 'node-2',
                'type': 'BatchNormalization',
                'params': {}
            },
            {
                'id': 'node-3',
                'type': 'ReLU',
                'params': {}
            },
            {
                'id': 'node-4',
                'type': 'MaxPooling2D',
                'params': {
                    'pool_size': [3, 3],
                    'strides': [2, 2]
                }
            },
            {
                'id': 'node-5',
                'type': 'GlobalAveragePooling2D',
                'params': {}
            },
            {
                'id': 'node-6',
                'type': 'Dense',
                'params': {
                    'units': 1000,
                    'activation': 'softmax'
                }
            }
        ]
    })
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'shapes' in data
