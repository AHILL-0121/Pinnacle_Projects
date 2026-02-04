"""
Chart and Graph Analyzer Service
Extracts insights from visual charts in financial documents
"""
import logging
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Types of charts detected"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    AREA = "area"
    STACKED_BAR = "stacked_bar"
    COMBO = "combo"
    WATERFALL = "waterfall"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    """Trend direction in chart"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class ChartInsight:
    """Insights extracted from a chart"""
    chart_type: ChartType
    title: Optional[str]
    trend: TrendDirection
    key_values: Dict[str, Any]
    insight: str
    page_number: int
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


class ChartAnalyzer:
    """
    Service for analyzing charts and graphs in financial documents.
    Uses Vision-Language Models for understanding chart content.
    Supports Google Gemini Vision.
    """
    
    def __init__(
        self, 
        vision_model: str = "gemini-1.5-pro",
        vision_provider: str = "gemini",
        gemini_api_key: Optional[str] = None
    ):
        self.vision_model = vision_model
        self.vision_provider = vision_provider.lower()
        self._gemini_available = False
        self._gemini_api_key = gemini_api_key
        self._init_vision_model()
    
    def _init_vision_model(self):
        """Initialize vision model client"""
        if self.vision_provider == "gemini":
            try:
                import google.generativeai as genai
                if self._gemini_api_key:
                    genai.configure(api_key=self._gemini_api_key)
                    self._gemini_client = genai.GenerativeModel(self.vision_model)
                    self._gemini_available = True
                    logger.info(f"Gemini Vision model initialized: {self.vision_model}")
                else:
                    logger.warning("Gemini API key not provided for vision")
            except Exception as e:
                logger.warning(f"Gemini Vision not available: {e}")
        else:
            logger.warning(f"Unknown vision provider: {self.vision_provider}. Only 'gemini' is supported.")
    
    def _vision_available(self) -> bool:
        """Check if vision model is available"""
        return self._gemini_available
    
    def analyze_charts(
        self, 
        image: Image.Image,
        page_number: int = 0,
        context: Optional[str] = None
    ) -> List[ChartInsight]:
        """
        Analyze charts in an image.
        
        Args:
            image: PIL Image containing charts
            page_number: Page number for reference
            context: Additional context about the document
            
        Returns:
            List of ChartInsight objects
        """
        insights = []
        
        # Step 1: Detect chart regions
        chart_regions = self._detect_chart_regions(image)
        
        if not chart_regions:
            # Analyze full image if no specific regions detected
            chart_regions = [(0, 0, image.width, image.height)]
        
        # Step 2: Analyze each chart region
        for bbox in chart_regions:
            # Crop chart region
            chart_image = image.crop(bbox)
            
            # Analyze with vision model
            if self._vision_available():
                insight = self._analyze_with_vision_model(
                    chart_image, 
                    page_number,
                    context
                )
                if insight:
                    insight.bbox = bbox
                    insights.append(insight)
            else:
                # Fallback to rule-based analysis
                insight = self._analyze_rule_based(chart_image, page_number)
                if insight:
                    insight.bbox = bbox
                    insights.append(insight)
        
        return insights
    
    def _detect_chart_regions(
        self, 
        image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions in image that likely contain charts.
        Uses color variance and shape detection heuristics.
        """
        img_array = np.array(image.convert('RGB'))
        height, width = img_array.shape[:2]
        
        regions = []
        
        # Divide image into grid cells
        grid_rows, grid_cols = 3, 2
        cell_height = height // grid_rows
        cell_width = width // grid_cols
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                y1 = i * cell_height
                y2 = min((i + 1) * cell_height, height)
                x1 = j * cell_width
                x2 = min((j + 1) * cell_width, width)
                
                cell = img_array[y1:y2, x1:x2]
                
                # Check if cell likely contains a chart
                if self._is_chart_region(cell):
                    regions.append((x1, y1, x2, y2))
        
        # Merge overlapping regions
        regions = self._merge_regions(regions)
        
        return regions
    
    def _is_chart_region(self, img_region: np.ndarray) -> bool:
        """Determine if image region likely contains a chart"""
        # Charts typically have:
        # 1. Multiple distinct colors
        # 2. Regular patterns/shapes
        # 3. Not predominantly text
        
        # Check color diversity
        unique_colors = len(np.unique(img_region.reshape(-1, 3), axis=0))
        color_ratio = unique_colors / (img_region.shape[0] * img_region.shape[1])
        
        # Moderate color diversity suggests chart (not too uniform, not too random)
        if 0.01 < color_ratio < 0.3:
            # Check for presence of saturated colors (common in charts)
            hsv_approx = np.std(img_region, axis=2)
            has_colored_regions = np.mean(hsv_approx > 30) > 0.1
            
            if has_colored_regions:
                return True
        
        return False
    
    def _merge_regions(
        self, 
        regions: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes"""
        if not regions:
            return []
        
        merged = []
        used = set()
        
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            
            current = list(r1)
            
            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # Check overlap
                if self._boxes_overlap(current, r2):
                    current = [
                        min(current[0], r2[0]),
                        min(current[1], r2[1]),
                        max(current[2], r2[2]),
                        max(current[3], r2[3])
                    ]
                    used.add(j)
            
            merged.append(tuple(current))
            used.add(i)
        
        return merged
    
    def _boxes_overlap(self, box1: List[int], box2: Tuple[int, ...]) -> bool:
        """Check if two boxes overlap"""
        return not (
            box1[2] < box2[0] or
            box1[0] > box2[2] or
            box1[3] < box2[1] or
            box1[1] > box2[3]
        )
    
    def _analyze_with_vision_model(
        self, 
        image: Image.Image,
        page_number: int,
        context: Optional[str]
    ) -> Optional[ChartInsight]:
        """Analyze chart using Vision model (OpenAI GPT-4V or Gemini Vision)"""
        
        # Build prompt
        prompt = """Analyze this financial chart/graph and extract:

1. Chart Type: (bar, line, pie, area, stacked_bar, combo, waterfall)
2. Title: (if visible)
3. Overall Trend: (up, down, stable, volatile)
4. Key Data Points: Extract approximate values for major data points
5. Financial Insight: Generate a one-sentence insight about what this chart shows

Focus on financial metrics like revenue, profit, growth rates, ratios, etc.

Respond in this exact JSON format:
{
    "chart_type": "bar|line|pie|area|stacked_bar|combo|waterfall|unknown",
    "title": "Chart title or null",
    "trend": "up|down|stable|volatile",
    "key_values": {"label1": value1, "label2": value2},
    "insight": "One sentence financial insight"
}"""

        if context:
            prompt += f"\n\nDocument context: {context}"
        
        try:
            content = None
            
            # Use Gemini vision
            if self._gemini_available:
                # Gemini vision with PIL image
                response = self._gemini_client.generate_content([prompt, image])
                content = response.text
            
            if not content:
                logger.warning("No response from vision model")
                return None
            
            # Parse response - extract JSON
            import json
            import re
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                
                return ChartInsight(
                    chart_type=ChartType(data.get('chart_type', 'unknown')),
                    title=data.get('title'),
                    trend=TrendDirection(data.get('trend', 'stable')),
                    key_values=data.get('key_values', {}),
                    insight=data.get('insight', ''),
                    page_number=page_number,
                    confidence=0.85
                )
            
        except Exception as e:
            logger.error(f"Vision model analysis failed: {e}")
        
        return None
    
    def _analyze_rule_based(
        self, 
        image: Image.Image,
        page_number: int
    ) -> Optional[ChartInsight]:
        """Fallback rule-based chart analysis"""
        img_array = np.array(image.convert('RGB'))
        
        # Detect chart type based on visual features
        chart_type = self._detect_chart_type(img_array)
        
        if chart_type == ChartType.UNKNOWN:
            return None
        
        # Estimate trend
        trend = self._estimate_trend(img_array, chart_type)
        
        # Generate generic insight
        insight = self._generate_basic_insight(chart_type, trend)
        
        return ChartInsight(
            chart_type=chart_type,
            title=None,
            trend=trend,
            key_values={},
            insight=insight,
            page_number=page_number,
            confidence=0.5
        )
    
    def _detect_chart_type(self, img_array: np.ndarray) -> ChartType:
        """Detect chart type from visual features"""
        height, width = img_array.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = np.mean(img_array, axis=2)
        
        # Simple edge detection
        h_edges = np.abs(np.diff(gray, axis=0))
        v_edges = np.abs(np.diff(gray, axis=1))
        
        # Count vertical and horizontal lines
        h_lines = np.sum(np.mean(h_edges, axis=1) > 20)
        v_lines = np.sum(np.mean(v_edges, axis=0) > 20)
        
        # Detect circles (for pie charts)
        # Simplified: check for color distribution from center
        center_y, center_x = height // 2, width // 2
        colors_near_center = img_array[
            center_y-10:center_y+10, 
            center_x-10:center_x+10
        ]
        color_variance_center = np.var(colors_near_center)
        
        # Heuristics
        if v_lines > h_lines * 2:
            return ChartType.BAR
        elif h_lines > v_lines:
            return ChartType.LINE
        elif color_variance_center < 100:
            return ChartType.PIE
        elif v_lines > 5 and h_lines > 5:
            return ChartType.COMBO
        
        return ChartType.UNKNOWN
    
    def _estimate_trend(
        self, 
        img_array: np.ndarray,
        chart_type: ChartType
    ) -> TrendDirection:
        """Estimate trend direction from chart image"""
        height, width = img_array.shape[:2]
        
        if chart_type == ChartType.PIE:
            return TrendDirection.STABLE
        
        # For bar/line charts, analyze left-to-right progression
        # Sample vertical positions of dark pixels (likely data points)
        gray = np.mean(img_array, axis=2)
        threshold = np.mean(gray) - np.std(gray)
        
        # Get average "height" of dark pixels in left and right halves
        left_half = gray[:, :width//3]
        right_half = gray[:, 2*width//3:]
        
        def get_centroid_y(region):
            mask = region < threshold
            if not np.any(mask):
                return region.shape[0] // 2
            y_coords = np.where(mask)[0]
            return np.mean(y_coords)
        
        left_y = get_centroid_y(left_half)
        right_y = get_centroid_y(right_half)
        
        # Note: In images, y increases downward
        diff = left_y - right_y
        
        if abs(diff) < height * 0.05:
            return TrendDirection.STABLE
        elif diff > 0:
            return TrendDirection.UP  # Data moved up (lower y)
        else:
            return TrendDirection.DOWN
    
    def _generate_basic_insight(
        self, 
        chart_type: ChartType,
        trend: TrendDirection
    ) -> str:
        """Generate basic insight from chart analysis"""
        insights = {
            (ChartType.BAR, TrendDirection.UP): "The metric shows an increasing trend across the period.",
            (ChartType.BAR, TrendDirection.DOWN): "The metric shows a declining trend across the period.",
            (ChartType.BAR, TrendDirection.STABLE): "The metric remains relatively stable across the period.",
            (ChartType.LINE, TrendDirection.UP): "The trend line indicates growth over time.",
            (ChartType.LINE, TrendDirection.DOWN): "The trend line indicates decline over time.",
            (ChartType.LINE, TrendDirection.STABLE): "The trend line shows relatively flat performance.",
            (ChartType.PIE, TrendDirection.STABLE): "The chart shows distribution of components.",
        }
        
        return insights.get(
            (chart_type, trend),
            f"Chart shows {chart_type.value} visualization with {trend.value} trend."
        )
    
    def generate_chart_summary(self, insights: List[ChartInsight]) -> str:
        """Generate a summary of all chart insights"""
        if not insights:
            return "No charts detected in the document."
        
        summary_parts = ["## Chart Analysis\n"]
        
        for i, insight in enumerate(insights, 1):
            parts = [f"### Chart {i}"]
            
            if insight.title:
                parts.append(f"**Title:** {insight.title}")
            
            parts.append(f"**Type:** {insight.chart_type.value.replace('_', ' ').title()}")
            parts.append(f"**Trend:** {self._trend_to_symbol(insight.trend)} {insight.trend.value.title()}")
            
            if insight.key_values:
                parts.append("**Key Values:**")
                for label, value in insight.key_values.items():
                    if isinstance(value, (int, float)):
                        parts.append(f"  - {label}: {value:,.2f}")
                    else:
                        parts.append(f"  - {label}: {value}")
            
            parts.append(f"**Insight:** {insight.insight}")
            parts.append("")
            
            summary_parts.append('\n'.join(parts))
        
        return '\n'.join(summary_parts)
    
    def _trend_to_symbol(self, trend: TrendDirection) -> str:
        """Convert trend to symbol"""
        symbols = {
            TrendDirection.UP: "↑",
            TrendDirection.DOWN: "↓",
            TrendDirection.STABLE: "→",
            TrendDirection.VOLATILE: "↕"
        }
        return symbols.get(trend, "→")
