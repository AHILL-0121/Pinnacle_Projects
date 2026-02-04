"""
Yahoo Finance service layer.
Handles all interactions with the yfinance library.
"""
import yfinance as yf
from datetime import datetime
from app.utils.errors import NotFoundError, UpstreamError


class YahooFinanceService:
    """Service for fetching data from Yahoo Finance."""
    
    @staticmethod
    def get_ticker(symbol):
        """
        Get yfinance Ticker object.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            yfinance Ticker object
        """
        try:
            return yf.Ticker(symbol)
        except Exception as e:
            raise UpstreamError(f"Failed to connect to Yahoo Finance: {str(e)}")
    
    @staticmethod
    def get_company_info(symbol):
        """
        Get company metadata.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
            
        Raises:
            NotFoundError: If symbol is invalid
            UpstreamError: If Yahoo Finance is unavailable
        """
        try:
            ticker = YahooFinanceService.get_ticker(symbol)
            info = ticker.info
            
            # Check if we got valid data
            if not info or info.get('regularMarketPrice') is None:
                # Try to verify if it's a valid symbol
                if info.get('symbol') is None and info.get('shortName') is None:
                    raise NotFoundError(f"Stock symbol '{symbol}' not found")
            
            # Extract key officers
            key_officers = []
            company_officers = info.get('companyOfficers', [])
            for officer in company_officers[:5]:  # Limit to top 5
                key_officers.append({
                    'name': officer.get('name', 'N/A'),
                    'title': officer.get('title', 'N/A')
                })
            
            return {
                'symbol': symbol.upper(),
                'name': info.get('longName') or info.get('shortName', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'business_summary': info.get('longBusinessSummary', 'N/A'),
                'key_officers': key_officers,
                'website': info.get('website', 'N/A'),
                'country': info.get('country', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A')
            }
            
        except NotFoundError:
            raise
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise NotFoundError(f"Stock symbol '{symbol}' not found")
            raise UpstreamError(f"Failed to fetch company data: {str(e)}")
    
    @staticmethod
    def get_stock_data(symbol):
        """
        Get real-time stock market data.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with current stock data
            
        Raises:
            NotFoundError: If symbol is invalid
            UpstreamError: If Yahoo Finance is unavailable
        """
        try:
            ticker = YahooFinanceService.get_ticker(symbol)
            info = ticker.info
            
            # Validate we have data
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if current_price is None:
                raise NotFoundError(f"Stock symbol '{symbol}' not found or no market data available")
            
            # Calculate change
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose', 0)
            change = round(current_price - prev_close, 2) if prev_close else 0
            change_percent = round((change / prev_close) * 100, 2) if prev_close else 0
            
            # Determine market state
            market_state = info.get('marketState', 'UNKNOWN')
            if market_state == 'REGULAR':
                market_state = 'OPEN'
            elif market_state in ['PRE', 'PREPRE']:
                market_state = 'PRE_MARKET'
            elif market_state in ['POST', 'POSTPOST']:
                market_state = 'AFTER_HOURS'
            elif market_state == 'CLOSED':
                market_state = 'CLOSED'
            
            return {
                'symbol': symbol.upper(),
                'market_state': market_state,
                'current_price': round(current_price, 2),
                'previous_close': round(prev_close, 2) if prev_close else None,
                'change': change,
                'change_percent': change_percent,
                'day_high': round(info.get('dayHigh', 0) or 0, 2),
                'day_low': round(info.get('dayLow', 0) or 0, 2),
                'open': round(info.get('open') or info.get('regularMarketOpen', 0) or 0, 2),
                'volume': info.get('volume') or info.get('regularMarketVolume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'fifty_two_week_high': round(info.get('fiftyTwoWeekHigh', 0) or 0, 2),
                'fifty_two_week_low': round(info.get('fiftyTwoWeekLow', 0) or 0, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except NotFoundError:
            raise
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise NotFoundError(f"Stock symbol '{symbol}' not found")
            raise UpstreamError(f"Failed to fetch stock data: {str(e)}")
    
    @staticmethod
    def get_historical_data(symbol, start_date, end_date, interval='1d'):
        """
        Get historical market data.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (datetime)
            end_date: End date (datetime)
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            Dictionary with historical data
            
        Raises:
            NotFoundError: If symbol is invalid or no data
            UpstreamError: If Yahoo Finance is unavailable
        """
        try:
            ticker = YahooFinanceService.get_ticker(symbol)
            
            # Fetch historical data
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            # Check if we got data
            if hist.empty:
                raise NotFoundError(
                    f"No historical data found for '{symbol}' in the specified date range"
                )
            
            # Convert to list of records
            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(row['Open'], 2) if row['Open'] else None,
                    'high': round(row['High'], 2) if row['High'] else None,
                    'low': round(row['Low'], 2) if row['Low'] else None,
                    'close': round(row['Close'], 2) if row['Close'] else None,
                    'volume': int(row['Volume']) if row['Volume'] else 0
                })
            
            return {
                'symbol': symbol.upper(),
                'interval': interval,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'records_count': len(data),
                'data': data
            }
            
        except NotFoundError:
            raise
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise NotFoundError(f"Stock symbol '{symbol}' not found")
            raise UpstreamError(f"Failed to fetch historical data: {str(e)}")


# Create singleton instance
yahoo_service = YahooFinanceService()
