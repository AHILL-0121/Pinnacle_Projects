"""
Analytical insights endpoint.
POST /api/analyze
"""
from flask import Blueprint, request, jsonify
from app.services.analysis_service import analysis_service
from app.utils.validators import validate_history_request


analysis_bp = Blueprint('analysis', __name__)


@analysis_bp.route('/analyze', methods=['POST'])
def analyze_stock():
    """
    Perform analytical insights on historical stock data.
    
    Request Body (JSON):
        {
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "interval": "1d"  // optional: 1d, 1wk, 1mo
        }
    
    Returns:
        JSON with analysis results:
        - average_price: Mean closing price
        - volatility: Annualized standard deviation (%)
        - trend: bullish/bearish/sideways/mildly_bullish/mildly_bearish
        - max_drawdown: Maximum peak-to-trough decline (%)
        - return_percent: Total return over period (%)
        - statistics: Additional stats (min, max, range, volume, etc.)
        - insight: Human-readable summary
    
    Errors:
        400: Invalid request body or insufficient data
        404: No data found for symbol/range
        503: Yahoo Finance unavailable
    """
    # Get and validate request data
    data = request.get_json(silent=True)
    validated = validate_history_request(data)
    
    # Perform analysis
    analysis_data = analysis_service.analyze_stock(
        symbol=validated['symbol'],
        start_date=validated['start_date'],
        end_date=validated['end_date'],
        interval=validated['interval']
    )
    
    return jsonify(analysis_data), 200
