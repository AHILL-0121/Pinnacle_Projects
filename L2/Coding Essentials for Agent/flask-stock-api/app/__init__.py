"""
Flask Stock Intelligence API
A production-ready REST API for stock market data analysis.
"""
from flask import Flask
from config import get_config


def create_app(config_class=None):
    """Application factory pattern."""
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)
    
    # Register blueprints
    from app.routes.company import company_bp
    from app.routes.stock import stock_bp
    from app.routes.history import history_bp
    from app.routes.analysis import analysis_bp
    
    app.register_blueprint(company_bp, url_prefix='/api')
    app.register_blueprint(stock_bp, url_prefix='/api')
    app.register_blueprint(history_bp, url_prefix='/api')
    app.register_blueprint(analysis_bp, url_prefix='/api')
    
    # Register error handlers
    from app.utils.errors import register_error_handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'Flask Stock Intelligence API'}
    
    return app
