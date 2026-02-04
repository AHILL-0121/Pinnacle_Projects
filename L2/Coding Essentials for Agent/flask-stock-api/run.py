"""
Flask Stock Intelligence API - Entry Point
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask Stock Intelligence API")
    print("="*50)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /api/company/<symbol> - Company info")
    print("  GET  /api/stock/<symbol>   - Real-time stock data")
    print("  POST /api/history          - Historical data")
    print("  POST /api/analyze          - Analytical insights")
    print("\n" + "="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
