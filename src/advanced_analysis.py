"""
Advanced analysis module for emotion, composition, style, and beauty filter detection.
"""

import cv2
import numpy as np
import math
from typing import Dict, Any, List


def analyze_emotion(image_array: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze facial expressions and emotions using geometric features."""
    if not faces:
        return {"emotion": "no_face", "confidence": 0, "emotion_score": 0}
    
    # Use the largest face
    main_face = max(faces, key=lambda f: f['face_ratio'])
    x, y, w, h = main_face['bbox']
    
    # Extract face region
    face_region = image_array[y:y+h, x:x+w]
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # Simple emotion detection based on face geometry
    # This is a basic implementation - more advanced would use deep learning
    
    # Analyze mouth curvature (smile detection)
    mouth_region = face_gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
    mouth_variance = np.var(mouth_region)
    
    # Analyze eye region (for squinting/happiness)
    eye_region = face_gray[int(h*0.25):int(h*0.5), int(w*0.2):int(w*0.8)]
    eye_mean = np.mean(eye_region)
    
    # Simple emotion classification based on features
    smile_score = min(100, mouth_variance / 10)
    eye_openness = min(100, eye_mean / 2)
    
    # Determine primary emotion
    if smile_score > 60 and eye_openness > 50:
        emotion = "happy"
        confidence = min(90, (smile_score + eye_openness) / 2)
    elif smile_score < 30 and eye_openness < 40:
        emotion = "serious"
        confidence = min(85, 100 - (smile_score + eye_openness) / 2)
    elif eye_openness > 70:
        emotion = "confident" 
        confidence = min(80, eye_openness)
    elif smile_score > 40:
        emotion = "pleasant"
        confidence = min(75, smile_score)
    else:
        emotion = "neutral"
        confidence = 60
    
    # Convert confidence to emotion score (higher = better for selfies)
    if emotion in ["happy", "confident", "pleasant"]:
        emotion_score = int(confidence)
    elif emotion == "neutral":
        emotion_score = int(confidence * 0.8)
    else:  # serious
        emotion_score = int(confidence * 0.6)
    
    return {
        "emotion": emotion,
        "confidence": int(confidence),
        "emotion_score": emotion_score
    }


def analyze_rule_of_thirds(image_array: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze composition using rule of thirds."""
    img_height, img_width = image_array.shape[:2]
    
    # Rule of thirds grid lines
    third_width = img_width / 3
    third_height = img_height / 3
    
    # Rule of thirds intersection points
    intersections = [
        (third_width, third_height),
        (2 * third_width, third_height),
        (third_width, 2 * third_height),
        (2 * third_width, 2 * third_height)
    ]
    
    if not faces:
        # Analyze general composition without faces
        # Convert to grayscale and find areas of interest
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Find edges/interest points
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate distribution of edges along grid lines
        edge_score = 60  # Default neutral score for non-face photos
        
        return {
            "composition_score": edge_score,
            "subject_placement": "no_clear_subject",
            "rule_of_thirds_score": edge_score
        }
    
    # Analyze face placement relative to rule of thirds
    main_face = max(faces, key=lambda f: f['face_ratio'])
    face_center = main_face['center']
    
    # Find closest intersection point
    min_distance = float('inf')
    closest_intersection = None
    
    for intersection in intersections:
        distance = math.sqrt(
            (face_center[0] - intersection[0])**2 + 
            (face_center[1] - intersection[1])**2
        )
        if distance < min_distance:
            min_distance = distance
            closest_intersection = intersection
    
    # Calculate composition score based on distance to nearest intersection
    max_distance = math.sqrt(img_width**2 + img_height**2) / 4
    distance_ratio = min_distance / max_distance
    
    composition_score = max(0, int(100 * (1 - distance_ratio)))
    
    # Determine placement quality
    if distance_ratio < 0.15:
        placement = "excellent"
    elif distance_ratio < 0.3:
        placement = "good"
    elif distance_ratio < 0.5:
        placement = "fair"
    else:
        placement = "centered"  # Often centered is not ideal for rule of thirds
    
    return {
        "composition_score": composition_score,
        "subject_placement": placement,
        "rule_of_thirds_score": composition_score,
        "distance_to_intersection": int(min_distance)
    }


def analyze_beauty_filter(image_array: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect over-processing or heavy beauty filters."""
    # This is a simplified detection - real filter detection is quite complex
    
    # Calculate overall image smoothness (over-smoothed skin indicator)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate local variance (smoothed areas have low variance)
    kernel = np.ones((5,5), np.float32) / 25
    smoothed = cv2.filter2D(gray, -1, kernel)
    variance_map = cv2.absdiff(gray, smoothed)
    avg_variance = np.mean(variance_map)
    
    # Calculate color saturation (over-saturated indicator)
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:,:,1])
    
    # Calculate edge sharpness (artificial sharpening indicator)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Detect artificial smoothing
    if avg_variance < 10:
        smoothing_level = "heavy"
        natural_score = 20
    elif avg_variance < 20:
        smoothing_level = "moderate" 
        natural_score = 50
    elif avg_variance < 30:
        smoothing_level = "light"
        natural_score = 75
    else:
        smoothing_level = "minimal"
        natural_score = 95
    
    # Detect over-saturation
    if saturation > 200:
        saturation_level = "over_saturated"
        sat_penalty = 30
    elif saturation > 150:
        saturation_level = "high"
        sat_penalty = 15
    else:
        saturation_level = "natural"
        sat_penalty = 0
    
    # Final naturalness score
    filter_score = max(10, natural_score - sat_penalty)
    
    # Determine if heavily filtered
    is_filtered = smoothing_level in ["heavy", "moderate"] or saturation_level == "over_saturated"
    
    return {
        "naturalness_score": filter_score,
        "smoothing_level": smoothing_level,
        "saturation_level": saturation_level,
        "is_heavily_filtered": is_filtered,
        "filter_detection_score": 100 - filter_score
    }


def analyze_outfit_style(image_array: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic clothing/style analysis using color and contrast patterns."""
    # This is a simplified analysis - advanced style detection would use deep learning
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Analyze overall color palette
    dominant_colors = []
    for i in range(3):  # RGB channels
        hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
        dominant_value = np.argmax(hist)
        dominant_colors.append(dominant_value)
    
    # Determine overall color scheme
    avg_brightness = np.mean(dominant_colors)
    color_variance = np.var(dominant_colors)
    
    # Classify style based on colors and patterns
    if avg_brightness > 200:
        color_scheme = "light"
        style_score = 75  # Light colors generally photograph well
    elif avg_brightness < 80:
        color_scheme = "dark"
        style_score = 65  # Dark colors can be harder to photograph
    else:
        color_scheme = "medium"
        style_score = 80
    
    # Analyze color coordination (low variance = more coordinated)
    if color_variance < 500:
        coordination = "coordinated"
        coord_bonus = 15
    elif color_variance < 2000:
        coordination = "somewhat_coordinated"
        coord_bonus = 5
    else:
        coordination = "varied"
        coord_bonus = -5
    
    # Calculate pattern complexity
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density < 0.05:
        pattern_type = "simple"
        pattern_score = 85  # Simple patterns photograph better
    elif edge_density < 0.15:
        pattern_type = "moderate"
        pattern_score = 75
    else:
        pattern_type = "complex"
        pattern_score = 60  # Complex patterns can be distracting
    
    # Overall style score
    final_style_score = max(20, min(100, style_score + coord_bonus + (pattern_score - 70)))
    
    # Determine style category
    if final_style_score > 80:
        style_category = "excellent"
    elif final_style_score > 65:
        style_category = "good"
    elif final_style_score > 45:
        style_category = "acceptable"
    else:
        style_category = "could_improve"
    
    return {
        "style_score": int(final_style_score),
        "color_scheme": color_scheme,
        "coordination": coordination,
        "pattern_type": pattern_type,
        "style_category": style_category
    }
