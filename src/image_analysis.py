"""
Basic image analysis module for technical image quality metrics.
"""

import cv2
import numpy as np
from typing import Dict, Any


def analyze_sharpness(image_array: np.ndarray) -> int:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize to 0-100 scale (typical range: 0-1000+)
    sharpness = int(np.clip(laplacian_var / 10, 0, 100))
    return sharpness


def analyze_lighting(image_array: np.ndarray) -> int:
    """Analyze lighting quality using histogram analysis."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Calculate histogram distribution
    total_pixels = gray.shape[0] * gray.shape[1]
    dark_pixels = np.sum(hist[0:85]) / total_pixels
    bright_pixels = np.sum(hist[170:256]) / total_pixels
    mid_pixels = np.sum(hist[85:170]) / total_pixels
    
    # Good lighting has balanced distribution
    balance_score = 1 - abs(dark_pixels - bright_pixels)
    mid_tone_bonus = min(mid_pixels * 2, 0.3)
    
    lighting_score = int((balance_score + mid_tone_bonus) * 100)
    return max(0, min(100, lighting_score))


def analyze_contrast(image_array: np.ndarray) -> int:
    """Analyze image contrast."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    contrast = gray.std()
    # Normalize to 0-100 scale (typical std range: 0-80)
    contrast_score = int(np.clip(contrast * 1.5, 0, 100))
    return contrast_score


def analyze_background_complexity(image_array: np.ndarray) -> int:
    """Analyze background clutter using edge density."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Lower edge density = less clutter = higher score
    clutter_score = int((1 - edge_density * 3) * 100)
    return max(0, min(100, clutter_score))


def analyze_color_harmony(image_array: np.ndarray) -> Dict[str, Any]:
    """Analyze color balance, temperature, and harmony."""
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Color temperature analysis (blue vs orange tones)
    r_mean = np.mean(image_array[:,:,0])
    g_mean = np.mean(image_array[:,:,1]) 
    b_mean = np.mean(image_array[:,:,2])
    
    # Calculate color temperature (simplified)
    if r_mean + g_mean > 0:
        warmth_ratio = (r_mean + g_mean) / (b_mean + 1)
    else:
        warmth_ratio = 1.0
    
    # Determine color temperature
    if warmth_ratio > 1.8:
        temperature = "warm"
        temp_score = min(85, warmth_ratio * 30)
    elif warmth_ratio < 0.8:
        temperature = "cool"
        temp_score = min(75, (1 / warmth_ratio) * 30)
    else:
        temperature = "neutral"
        temp_score = 90
    
    # Color saturation analysis
    saturation = np.mean(hsv[:,:,1])
    sat_score = min(100, saturation / 2.55)
    
    # Color distribution (how varied the colors are)
    hue_std = np.std(hsv[:,:,0])
    color_variety = min(100, hue_std * 2)
    
    # Overall color harmony score
    harmony_score = int((temp_score + sat_score + color_variety) / 3)
    
    return {
        "color_temperature": temperature,
        "temperature_score": int(temp_score),
        "saturation_score": int(sat_score),
        "color_variety": int(color_variety),
        "color_harmony_score": harmony_score
    }


def analyze_time_of_day(image_array: np.ndarray) -> Dict[str, Any]:
    """Identify lighting conditions and time of day characteristics."""
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Analyze color temperature and lighting characteristics
    r_mean = np.mean(image_array[:,:,0])
    g_mean = np.mean(image_array[:,:,1])
    b_mean = np.mean(image_array[:,:,2])
    
    # Overall brightness
    brightness = np.mean([r_mean, g_mean, b_mean])
    
    # Color temperature analysis
    if r_mean > g_mean and r_mean > b_mean:
        if (r_mean - b_mean) > 50:
            warmth_level = "very_warm"
        else:
            warmth_level = "warm"
    elif b_mean > r_mean and b_mean > g_mean:
        if (b_mean - r_mean) > 30:
            warmth_level = "very_cool"
        else:
            warmth_level = "cool"
    else:
        warmth_level = "neutral"
    
    # Determine lighting type and time
    if brightness > 180 and warmth_level in ["neutral", "cool"]:
        lighting_type = "bright_daylight"
        time_category = "midday"
        quality_score = 85
    elif brightness > 120 and warmth_level == "warm":
        lighting_type = "golden_hour"
        time_category = "sunrise_sunset"
        quality_score = 95  # Golden hour is ideal
    elif brightness > 100 and warmth_level == "very_warm":
        lighting_type = "warm_indoor"
        time_category = "indoor_warm"
        quality_score = 70
    elif brightness < 80 and warmth_level == "cool":
        lighting_type = "blue_hour"
        time_category = "twilight"
        quality_score = 75
    elif brightness < 60:
        lighting_type = "low_light"
        time_category = "evening_night"
        quality_score = 45
    else:
        lighting_type = "mixed"
        time_category = "varied"
        quality_score = 60
    
    # Additional lighting quality factors
    contrast = np.std(image_array)
    if contrast > 40:
        contrast_quality = "good"
        contrast_bonus = 10
    elif contrast > 20:
        contrast_quality = "moderate"
        contrast_bonus = 0
    else:
        contrast_quality = "low"
        contrast_bonus = -10
    
    # Final lighting quality score
    final_lighting_score = max(20, min(100, quality_score + contrast_bonus))
    
    return {
        "lighting_quality_score": int(final_lighting_score),
        "lighting_type": lighting_type,
        "time_category": time_category,
        "warmth_level": warmth_level,
        "brightness_level": int(brightness),
        "contrast_quality": contrast_quality
    }
