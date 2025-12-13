"""
Configuration for the No-Code Interface
"""

import os


class Config:
    """Base configuration"""
    
    # Server settings
    HOST = os.getenv('NOCODE_HOST', '127.0.0.1')
    PORT = int(os.getenv('NOCODE_PORT', 8051))
    DEBUG = os.getenv('NOCODE_DEBUG', 'False').lower() == 'true'
    
    # Paths
    BASE_DIR = os.path.dirname(__file__)
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    EXPORTED_MODELS_DIR = os.path.join(BASE_DIR, 'exported_models')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    
    # UI settings
    DEFAULT_INPUT_SHAPE = [None, 28, 28, 1]
    DEFAULT_OPTIMIZER = {
        'type': 'Adam',
        'params': {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999
        }
    }
    DEFAULT_LOSS = 'categorical_crossentropy'
    
    # Validation settings
    MAX_LAYERS = 100
    VALIDATION_TIMEOUT = 10
    
    # Code generation settings
    SUPPORTED_BACKENDS = ['tensorflow', 'pytorch', 'onnx']
    DEFAULT_BACKEND = 'tensorflow'
    
    # Feature flags
    ENABLE_TUTORIALS = True
    ENABLE_TEMPLATES = True
    ENABLE_VALIDATION = True
    ENABLE_CODE_GENERATION = True
    ENABLE_MODEL_SAVE = True
    
    # Security settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = {'.json', '.neural'}
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        os.makedirs(cls.SAVED_MODELS_DIR, exist_ok=True)
        os.makedirs(cls.EXPORTED_MODELS_DIR, exist_ok=True)
        os.makedirs(cls.TEMPLATES_DIR, exist_ok=True)
        os.makedirs(cls.STATIC_DIR, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    CORS_ORIGINS = ['*']


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SAVED_MODELS_DIR = '/tmp/neural_nocode_test/saved_models'
    EXPORTED_MODELS_DIR = '/tmp/neural_nocode_test/exported_models'


configs = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration for environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    config_class = configs.get(env, DevelopmentConfig)
    config_class.ensure_directories()
    
    return config_class
