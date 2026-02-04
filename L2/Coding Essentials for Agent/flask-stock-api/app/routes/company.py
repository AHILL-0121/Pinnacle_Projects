"""
Company information endpoint.
GET /api/company/<symbol>
"""
from flask import Blueprint, jsonify
from app.services.yahoo_service import yahoo_service
from app.utils.validators import validate_symbol
from app.utils.errors import success_response


company_bp = Blueprint('company', __name__)


@company_bp.route('/company/<symbol>', methods=['GET'])
def get_company_info(symbol):
    """
    Get company metadata.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    
    Returns:
        JSON with company information including:
        - name, industry, sector
        - business summary
        - key officers
        - additional metadata
    
    Errors:
        404: Invalid or unknown symbol
        503: Yahoo Finance unavailable
    """
    # Validate symbol
    symbol = validate_symbol(symbol)
    
    # Fetch company data
    company_data = yahoo_service.get_company_info(symbol)
    
    return jsonify(company_data), 200
