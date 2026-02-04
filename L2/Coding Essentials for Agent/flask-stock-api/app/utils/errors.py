"""
Custom error classes and handlers for the Stock API.
"""
from flask import jsonify
from werkzeug.exceptions import HTTPException


class APIError(Exception):
    """Base API Error class."""
    
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        """Convert error to dictionary."""
        rv = dict(self.payload or ())
        rv['error'] = True
        rv['message'] = self.message
        rv['status_code'] = self.status_code
        return rv


class ValidationError(APIError):
    """Validation error - 400 Bad Request."""
    
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=400, payload=payload)


class NotFoundError(APIError):
    """Resource not found - 404."""
    
    def __init__(self, message="Resource not found", payload=None):
        super().__init__(message, status_code=404, payload=payload)


class UpstreamError(APIError):
    """Upstream service unavailable - 503."""
    
    def __init__(self, message="Upstream data service unavailable", payload=None):
        super().__init__(message, status_code=503, payload=payload)


class RateLimitError(APIError):
    """Rate limit exceeded - 429."""
    
    def __init__(self, message="Rate limit exceeded. Please try again later.", payload=None):
        super().__init__(message, status_code=429, payload=payload)


def register_error_handlers(app):
    """Register error handlers with the Flask app."""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """Handle custom API errors."""
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle standard HTTP exceptions."""
        response = jsonify({
            'error': True,
            'message': error.description,
            'status_code': error.code
        })
        response.status_code = error.code
        return response
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle unexpected exceptions."""
        # Log the error in production
        app.logger.error(f"Unexpected error: {str(error)}")
        
        response = jsonify({
            'error': True,
            'message': 'An unexpected error occurred',
            'status_code': 500
        })
        response.status_code = 500
        return response


def error_response(message, status_code=400, **kwargs):
    """Create a standardized error response."""
    response_data = {
        'error': True,
        'message': message,
        'status_code': status_code,
        **kwargs
    }
    return jsonify(response_data), status_code


def success_response(data, status_code=200):
    """Create a standardized success response."""
    return jsonify(data), status_code
