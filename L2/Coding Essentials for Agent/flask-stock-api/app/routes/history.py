"""
Historical market data endpoint.
POST /api/history
"""
from flask import Blueprint, request, jsonify
from app.services.yahoo_service import yahoo_service
from app.utils.validators import validate_history_request


history_bp = Blueprint('history', __name__)


@history_bp.route('/history', methods=['POST'])
def get_historical_data():
    """
    Get historical market data for a stock.
    
    Request Body (JSON):
        {
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "interval": "1d"  // optional: 1d, 1wk, 1mo
        }
    
    Returns:
        JSON with historical OHLCV data:
        - symbol, interval, date range
        - array of data points with date, open, high, low, close, volume
    
    Errors:
        400: Invalid request body or parameters
        404: No data found for symbol/range
        503: Yahoo Finance unavailable
    """
    # Get and validate request data
    data = request.get_json(silent=True)
    validated = validate_history_request(data)
    
    # Fetch historical data
    history_data = yahoo_service.get_historical_data(
        symbol=validated['symbol'],
        start_date=validated['start_date'],
        end_date=validated['end_date'],
        interval=validated['interval']
    )
    
    return jsonify(history_data), 200
