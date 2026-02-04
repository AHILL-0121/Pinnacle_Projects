"""
Stock market data endpoint.
GET /api/stock/<symbol>
"""
from flask import Blueprint, jsonify
from app.services.yahoo_service import yahoo_service
from app.utils.validators import validate_symbol


stock_bp = Blueprint('stock', __name__)


@stock_bp.route('/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """
    Get real-time stock market data.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    
    Returns:
        JSON with current stock data including:
        - current price
        - day's change (amount and percent)
        - day high/low
        - volume
        - market state
    
    Errors:
        404: Invalid or unknown symbol
        503: Yahoo Finance unavailable
    """
    # Validate symbol
    symbol = validate_symbol(symbol)
    
    # Fetch stock data
    stock_data = yahoo_service.get_stock_data(symbol)
    
    return jsonify(stock_data), 200
