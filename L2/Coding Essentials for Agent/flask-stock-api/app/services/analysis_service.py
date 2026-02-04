"""
Analysis service layer.
Performs analytical computations on stock data.
"""
import numpy as np
import pandas as pd
from app.services.yahoo_service import yahoo_service
from app.utils.errors import NotFoundError, ValidationError


class AnalysisService:
    """Service for performing stock analysis."""
    
    @staticmethod
    def analyze_stock(symbol, start_date, end_date, interval='1d'):
        """
        Perform comprehensive analysis on historical stock data.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (datetime)
            end_date: End date (datetime)
            interval: Data interval
            
        Returns:
            Dictionary with analysis results
        """
        # Get historical data
        history = yahoo_service.get_historical_data(symbol, start_date, end_date, interval)
        data = history['data']
        
        if len(data) < 2:
            raise ValidationError("Insufficient data for analysis. Need at least 2 data points.")
        
        # Extract closing prices
        closes = [d['close'] for d in data if d['close'] is not None]
        highs = [d['high'] for d in data if d['high'] is not None]
        lows = [d['low'] for d in data if d['low'] is not None]
        volumes = [d['volume'] for d in data if d['volume'] is not None]
        
        if len(closes) < 2:
            raise ValidationError("Insufficient closing price data for analysis.")
        
        closes_array = np.array(closes)
        
        # Calculate metrics
        average_price = AnalysisService._calculate_average(closes_array)
        volatility = AnalysisService._calculate_volatility(closes_array)
        trend = AnalysisService._detect_trend(closes_array)
        max_drawdown = AnalysisService._calculate_max_drawdown(closes_array)
        return_percent = AnalysisService._calculate_return(closes_array)
        
        # Additional metrics
        avg_volume = np.mean(volumes) if volumes else 0
        price_range = max(highs) - min(lows) if highs and lows else 0
        
        # Generate insight
        insight = AnalysisService._generate_insight(
            trend, volatility, return_percent, max_drawdown
        )
        
        return {
            'symbol': symbol.upper(),
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'data_points': len(closes),
            'average_price': round(average_price, 2),
            'volatility': round(volatility, 2),
            'trend': trend,
            'max_drawdown': round(max_drawdown, 2),
            'return_percent': round(return_percent, 2),
            'statistics': {
                'min_price': round(min(closes), 2),
                'max_price': round(max(closes), 2),
                'price_range': round(price_range, 2),
                'avg_volume': int(avg_volume),
                'start_price': round(closes[0], 2),
                'end_price': round(closes[-1], 2)
            },
            'insight': insight
        }
    
    @staticmethod
    def _calculate_average(prices):
        """Calculate average price."""
        return np.mean(prices)
    
    @staticmethod
    def _calculate_volatility(prices):
        """
        Calculate volatility as standard deviation of daily returns.
        Annualized for daily data.
        """
        if len(prices) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = np.diff(prices) / prices[:-1]
        
        # Standard deviation of returns
        daily_volatility = np.std(returns)
        
        # Annualize (assuming ~252 trading days)
        annualized_volatility = daily_volatility * np.sqrt(252) * 100
        
        return annualized_volatility
    
    @staticmethod
    def _detect_trend(prices):
        """
        Detect overall price trend.
        Uses linear regression slope and price comparison.
        """
        if len(prices) < 2:
            return 'sideways'
        
        # Method 1: Simple start-end comparison
        start_price = prices[0]
        end_price = prices[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        
        # Method 2: Linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        slope_percent = (slope / np.mean(prices)) * 100
        
        # Combine signals
        if percent_change > 5 and slope_percent > 0:
            return 'bullish'
        elif percent_change < -5 and slope_percent < 0:
            return 'bearish'
        elif abs(percent_change) <= 5:
            return 'sideways'
        elif percent_change > 0:
            return 'mildly_bullish'
        else:
            return 'mildly_bearish'
    
    @staticmethod
    def _calculate_max_drawdown(prices):
        """
        Calculate maximum drawdown percentage.
        Maximum peak-to-trough decline.
        """
        if len(prices) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdown at each point
        drawdowns = (prices - running_max) / running_max * 100
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        return max_drawdown
    
    @staticmethod
    def _calculate_return(prices):
        """Calculate total return percentage."""
        if len(prices) < 2:
            return 0.0
        
        start_price = prices[0]
        end_price = prices[-1]
        
        return ((end_price - start_price) / start_price) * 100
    
    @staticmethod
    def _generate_insight(trend, volatility, return_percent, max_drawdown):
        """Generate human-readable insight based on analysis."""
        insights = []
        
        # Trend insight
        if trend == 'bullish':
            insights.append("The stock shows a strong upward trend")
        elif trend == 'bearish':
            insights.append("The stock shows a significant downward trend")
        elif trend == 'mildly_bullish':
            insights.append("The stock shows a slight upward movement")
        elif trend == 'mildly_bearish':
            insights.append("The stock shows a slight downward movement")
        else:
            insights.append("The stock has been trading sideways")
        
        # Volatility insight
        if volatility < 15:
            insights.append("with low volatility, indicating stable price action")
        elif volatility < 30:
            insights.append("with moderate volatility")
        else:
            insights.append("with high volatility, indicating significant price swings")
        
        # Return insight
        if abs(return_percent) > 20:
            if return_percent > 0:
                insights.append(f"The {return_percent:.1f}% gain is substantial")
            else:
                insights.append(f"The {abs(return_percent):.1f}% loss is significant")
        
        # Drawdown warning
        if max_drawdown < -15:
            insights.append(f"Note: Maximum drawdown of {abs(max_drawdown):.1f}% suggests notable risk")
        
        return ". ".join(insights) + "."
    
    @staticmethod
    def calculate_moving_averages(prices, windows=[20, 50, 200]):
        """
        Calculate moving averages for given windows.
        
        Args:
            prices: List of prices
            windows: List of window sizes
            
        Returns:
            Dictionary with moving averages
        """
        result = {}
        prices_series = pd.Series(prices)
        
        for window in windows:
            if len(prices) >= window:
                ma = prices_series.rolling(window=window).mean()
                result[f'ma_{window}'] = round(ma.iloc[-1], 2)
            else:
                result[f'ma_{window}'] = None
        
        return result
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: List of prices
            period: RSI period (default 14)
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) <= period:
            return None
        
        prices_series = pd.Series(prices)
        delta = prices_series.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi.iloc[-1], 2)


# Create singleton instance
analysis_service = AnalysisService()
