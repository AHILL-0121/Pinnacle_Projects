"""
Routes package initialization.
"""
from app.routes.company import company_bp
from app.routes.stock import stock_bp
from app.routes.history import history_bp
from app.routes.analysis import analysis_bp

__all__ = [
    'company_bp',
    'stock_bp',
    'history_bp',
    'analysis_bp'
]
