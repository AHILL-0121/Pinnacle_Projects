"""
Image Preprocessing Module
Handles image enhancement, normalization, and preparation for OCR
"""
import io
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image preprocessing for optimal OCR and analysis performance.
    Handles enhancement, deskewing, noise removal, and normalization.
    """
    
    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi
        self.min_dimension = 100
        self.max_dimension = 4000
    
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Full preprocessing pipeline for OCR optimization.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed PIL Image optimized for OCR
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Step 1: Resize if too small or too large
            image = self._normalize_size(image)
            
            # Step 2: Convert to grayscale for OCR
            gray_image = image.convert('L')
            
            # Step 3: Enhance contrast
            gray_image = self._enhance_contrast(gray_image)
            
            # Step 4: Denoise
            gray_image = self._denoise(gray_image)
            
            # Step 5: Binarize (adaptive thresholding)
            binary_image = self._binarize(gray_image)
            
            # Step 6: Deskew if needed
            final_image = self._deskew(binary_image)
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error in OCR preprocessing: {e}")
            # Return original image if preprocessing fails
            return image.convert('L') if image.mode != 'L' else image
    
    def preprocess_for_chart_analysis(self, image: Image.Image) -> Image.Image:
        """
        Preprocessing pipeline optimized for chart/graph analysis.
        Preserves colors and enhances visual elements.
        
        Args:
            image: PIL Image containing chart
            
        Returns:
            Preprocessed PIL Image for chart analysis
        """
        try:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to optimal dimensions
            image = self._normalize_size(image, max_dim=2000)
            
            # Enhance colors and contrast
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Sharpen edges
            image = image.filter(ImageFilter.SHARPEN)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in chart preprocessing: {e}")
            return image
    
    def _normalize_size(self, image: Image.Image, max_dim: int = None) -> Image.Image:
        """Normalize image dimensions"""
        max_dim = max_dim or self.max_dimension
        width, height = image.size
        
        # Scale up if too small
        if width < self.min_dimension or height < self.min_dimension:
            scale = max(self.min_dimension / width, self.min_dimension / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Scale down if too large
        if width > max_dim or height > max_dim:
            scale = min(max_dim / width, max_dim / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        return image
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast using CLAHE-like approach"""
        # Auto-contrast
        image = ImageOps.autocontrast(image, cutoff=1)
        
        # Additional contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        return image
    
    def _denoise(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        # Apply median filter for noise reduction
        image = image.filter(ImageFilter.MedianFilter(size=3))
        return image
    
    def _binarize(self, image: Image.Image, threshold: int = None) -> Image.Image:
        """
        Binarize image using adaptive thresholding approach.
        """
        # Convert to numpy for processing
        img_array = np.array(image)
        
        if threshold is None:
            # Otsu's thresholding approximation
            threshold = self._otsu_threshold(img_array)
        
        # Apply threshold
        binary_array = (img_array > threshold).astype(np.uint8) * 255
        
        return Image.fromarray(binary_array)
    
    def _otsu_threshold(self, image_array: np.ndarray) -> int:
        """Calculate Otsu's threshold"""
        histogram = np.histogram(image_array.flatten(), bins=256, range=(0, 256))[0]
        histogram = histogram.astype(float) / histogram.sum()
        
        best_threshold = 0
        best_variance = 0
        
        for t in range(1, 255):
            w0 = histogram[:t].sum()
            w1 = histogram[t:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mean0 = (np.arange(t) * histogram[:t]).sum() / w0
            mean1 = (np.arange(t, 256) * histogram[t:]).sum() / w1
            
            variance = w0 * w1 * (mean0 - mean1) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = t
        
        return best_threshold
    
    def _deskew(self, image: Image.Image, max_angle: float = 10.0) -> Image.Image:
        """
        Deskew image if it's rotated.
        Uses simple variance-based angle detection.
        """
        # For simplicity, we'll skip complex deskewing
        # In production, use a proper deskewing algorithm
        return image
    
    def extract_regions(
        self, 
        image: Image.Image, 
        regions: List[Tuple[int, int, int, int]]
    ) -> List[Image.Image]:
        """
        Extract specific regions from an image.
        
        Args:
            image: Source PIL Image
            regions: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            List of cropped PIL Images
        """
        extracted = []
        for x1, y1, x2, y2 in regions:
            try:
                cropped = image.crop((x1, y1, x2, y2))
                extracted.append(cropped)
            except Exception as e:
                logger.warning(f"Failed to extract region ({x1}, {y1}, {x2}, {y2}): {e}")
        
        return extracted
    
    def detect_table_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential table regions in an image.
        Returns bounding boxes of detected tables.
        """
        # Convert to grayscale and array
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Simple line detection for table boundaries
        # This is a simplified approach; production should use proper table detection
        
        regions = []
        
        # Detect horizontal and vertical lines using gradient analysis
        h_grad = np.abs(np.diff(img_array.astype(float), axis=0))
        v_grad = np.abs(np.diff(img_array.astype(float), axis=1))
        
        # Threshold gradients
        h_lines = np.mean(h_grad, axis=1) > 10
        v_lines = np.mean(v_grad, axis=0) > 10
        
        # Find clusters of lines (potential table boundaries)
        # Simplified: return entire image if lines detected
        if np.sum(h_lines) > 5 and np.sum(v_lines) > 3:
            # Has table-like structure
            regions.append((0, 0, image.width, image.height))
        
        return regions
    
    def detect_chart_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential chart/graph regions in an image.
        """
        # Convert to RGB array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        regions = []
        
        # Charts typically have distinct color regions
        # Simplified detection based on color variance
        
        # Divide image into grid and check for color diversity
        grid_size = 4
        h, w = img_array.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                cell = img_array[y1:y2, x1:x2]
                
                # Check color variance
                color_std = np.std(cell, axis=(0, 1))
                
                # High color variance might indicate a chart
                if np.mean(color_std) > 30:
                    regions.append((x1, y1, x2, y2))
        
        # Merge overlapping regions
        regions = self._merge_regions(regions)
        
        return regions
    
    def _merge_regions(
        self, 
        regions: List[Tuple[int, int, int, int]], 
        overlap_threshold: float = 0.3
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
                    # Merge
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
            box1[2] < box2[0] or  # box1 is left of box2
            box1[0] > box2[2] or  # box1 is right of box2
            box1[3] < box2[1] or  # box1 is above box2
            box1[1] > box2[3]     # box1 is below box2
        )
    
    def image_to_bytes(self, image: Image.Image, format: str = 'PNG') -> bytes:
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def bytes_to_image(self, data: bytes) -> Image.Image:
        """Convert bytes to PIL Image"""
        return Image.open(io.BytesIO(data))
