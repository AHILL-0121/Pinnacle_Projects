"""
Flask Stock Intelligence API - Configuration
"""
import os


class Config:
    """Base configuration."""
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False
    
    # Yahoo Finance settings
    YAHOO_TIMEOUT = 10  # seconds
    
    # Validation settings
    VALID_INTERVALS = ['1d', '1wk', '1mo']
    MAX_DATE_RANGE_DAYS = 3650  # ~10 years max
    
    # Response settings
    MAX_HISTORY_RECORDS = 5000


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)
