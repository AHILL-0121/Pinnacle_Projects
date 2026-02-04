"""
Input validation utilities for the Stock API.
"""
import re
from datetime import datetime, timedelta
from app.utils.errors import ValidationError


# Valid intervals for historical data
VALID_INTERVALS = ['1d', '1wk', '1mo']

# Stock symbol pattern (alphanumeric, 1-10 characters, may include dots and hyphens)
SYMBOL_PATTERN = re.compile(r'^[A-Za-z0-9.\-]{1,10}$')


def validate_symbol(symbol):
    """
    Validate stock ticker symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Cleaned uppercase symbol
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Stock symbol is required")
    
    symbol = str(symbol).strip().upper()
    
    if not SYMBOL_PATTERN.match(symbol):
        raise ValidationError(
            f"Invalid stock symbol '{symbol}'. "
            "Symbol must be 1-10 alphanumeric characters."
        )
    
    return symbol


def validate_date(date_string, field_name="date"):
    """
    Validate and parse date string.
    
    Args:
        date_string: Date in YYYY-MM-DD format
        field_name: Name of the field for error messages
        
    Returns:
        datetime object
        
    Raises:
        ValidationError: If date is invalid
    """
    if not date_string:
        raise ValidationError(f"{field_name} is required")
    
    try:
        return datetime.strptime(str(date_string).strip(), '%Y-%m-%d')
    except ValueError:
        raise ValidationError(
            f"Invalid {field_name} format. Use YYYY-MM-DD (e.g., 2024-01-01)"
        )


def validate_date_range(start_date, end_date, max_days=3650):
    """
    Validate date range for historical data.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        max_days: Maximum allowed range in days
        
    Returns:
        Tuple of (start_datetime, end_datetime)
        
    Raises:
        ValidationError: If date range is invalid
    """
    start_dt = validate_date(start_date, "start_date")
    end_dt = validate_date(end_date, "end_date")
    
    # Check start < end
    if start_dt >= end_dt:
        raise ValidationError("start_date must be before end_date")
    
    # Check not in future
    today = datetime.now()
    if start_dt > today:
        raise ValidationError("start_date cannot be in the future")
    
    # Check range not too large
    delta = (end_dt - start_dt).days
    if delta > max_days:
        raise ValidationError(
            f"Date range exceeds maximum of {max_days} days ({delta} days requested)"
        )
    
    return start_dt, end_dt


def validate_interval(interval):
    """
    Validate data interval.
    
    Args:
        interval: Data interval string
        
    Returns:
        Validated interval string
        
    Raises:
        ValidationError: If interval is invalid
    """
    if not interval:
        return '1d'  # Default to daily
    
    interval = str(interval).strip().lower()
    
    if interval not in VALID_INTERVALS:
        raise ValidationError(
            f"Invalid interval '{interval}'. "
            f"Must be one of: {', '.join(VALID_INTERVALS)}"
        )
    
    return interval


def validate_history_request(data):
    """
    Validate complete history/analysis request.
    
    Args:
        data: Request JSON data
        
    Returns:
        Dictionary with validated fields
        
    Raises:
        ValidationError: If any field is invalid
    """
    if not data:
        raise ValidationError("Request body is required")
    
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    
    # Validate symbol
    if 'symbol' not in data:
        raise ValidationError("'symbol' field is required")
    symbol = validate_symbol(data['symbol'])
    
    # Validate dates
    if 'start_date' not in data:
        raise ValidationError("'start_date' field is required")
    if 'end_date' not in data:
        raise ValidationError("'end_date' field is required")
    
    start_dt, end_dt = validate_date_range(data['start_date'], data['end_date'])
    
    # Validate interval
    interval = validate_interval(data.get('interval', '1d'))
    
    return {
        'symbol': symbol,
        'start_date': start_dt,
        'end_date': end_dt,
        'interval': interval
    }


def sanitize_string(value, max_length=1000):
    """
    Sanitize string input to prevent injection.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if value is None:
        return None
    
    value = str(value).strip()
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]
    
    # Remove potentially dangerous characters
    value = re.sub(r'[<>{}]', '', value)
    
    return value
