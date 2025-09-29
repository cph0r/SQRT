import gradio as gr
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import json
import cv2
import csv
import io
import datetime
import os
import math

# Fix Gradio Python 3.13 compatibility issue
def patched_get_api_info(self):
    return {"named_endpoints": {}, "unnamed_endpoints": {}}

# Apply the patch
gr.blocks.Blocks.get_api_info = patched_get_api_info

# Workaround: bypass Gradio API schema generation that crashes on Python 3.13
try:
    import gradio.blocks as _gr_blocks  # type: ignore
    if hasattr(_gr_blocks, "Blocks") and hasattr(_gr_blocks.Blocks, "get_api_info"):
        _gr_blocks.Blocks.get_api_info = lambda self: {
            "named_endpoints": {},
            "unnamed_endpoints": {}
        }
except Exception:
    pass

# Additional workaround for Python 3.13 compatibility
try:
    import gradio_client.utils as _gr_utils  # type: ignore
    original_get_type = _gr_utils.get_type
    
    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return original_get_type(schema)
    
    _gr_utils.get_type = patched_get_type
except Exception:
    pass


def detect_faces(image_array: np.ndarray) -> List[Dict[str, Any]]:
    """Detect faces in the image and return face information."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    img_height, img_width = image_array.shape[:2]
    
    # Load OpenCV's pre-trained face classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Try frontal face detection first
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,          # Moderate steps between scales
        minNeighbors=6,           # Moderate requirement for neighboring detections
        minSize=(40, 40),         # Reasonable minimum face size
        maxSize=(300, 300),       # Reasonable maximum size limit
        flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
    )
    
    # If no frontal faces found, try profile detection
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(40, 40),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
    
    # Aggressive filtering to remove false positives
    if len(faces) > 0:
        # First, filter by size relative to image
        img_area = img_height * img_width
        size_filtered = []
        
        for (x, y, w, h) in faces:
            face_area = w * h
            face_ratio = face_area / img_area
            
            # Only keep faces that are a reasonable size (0.2% to 30% of image)
            if 0.002 <= face_ratio <= 0.30:
                size_filtered.append((x, y, w, h))
        
        faces = size_filtered
        
        # Then remove overlapping detections more aggressively
        if len(faces) > 1:
            filtered_faces = []
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # Sort by area, largest first
            
            for i, (x1, y1, w1, h1) in enumerate(faces_sorted):
                area1 = w1 * h1
                keep = True
                
                for j, (x2, y2, w2, h2) in enumerate(faces_sorted[:i]):  # Only check against already accepted faces
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    # If any overlap > 10%, remove the smaller face (more aggressive)
                    if overlap_area > 0.10 * area1:
                        keep = False
                        break
                
                if keep:
                    filtered_faces.append((x1, y1, w1, h1))
            
            faces = filtered_faces
        
        # Final check: if we still have multiple faces, keep only the largest one
        # This is reasonable for selfies which typically have one main subject
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]
    
    face_data = []
    
    for (x, y, w, h) in faces:
        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate image center
        img_center_x = img_width // 2
        img_center_y = img_height // 2
        
        # Calculate positioning scores
        horizontal_offset = abs(face_center_x - img_center_x) / (img_width / 2)
        vertical_offset = abs(face_center_y - img_center_y) / (img_height / 2)
        
        # Face size relative to image
        face_area = w * h
        image_area = img_width * img_height
        face_ratio = face_area / image_area
        
        face_data.append({
            'bbox': (x, y, w, h),
            'center': (face_center_x, face_center_y),
            'horizontal_offset': horizontal_offset,
            'vertical_offset': vertical_offset,
            'face_ratio': face_ratio,
            'size_score': min(100, max(0, (face_ratio - 0.05) * 500)),  # Optimal face size around 10-30% of image
            'centering_score': max(0, 100 - (horizontal_offset + vertical_offset) * 100)
        })
    
    return face_data

def analyze_face_quality(image_array: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze face-specific quality metrics."""
    if not faces:
        return {
            "face_detection": 0,
            "face_positioning": 0,
            "face_size": 0,
            "face_lighting": 0
        }
    
    # Use the largest/most prominent face
    main_face = max(faces, key=lambda f: f['face_ratio'])
    x, y, w, h = main_face['bbox']
    
    # Extract face region
    face_region = image_array[y:y+h, x:x+w]
    
    # Analyze face lighting
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    face_brightness = np.mean(face_gray)
    face_contrast = np.std(face_gray)
    
    # Face lighting score (optimal brightness around 120-180)
    brightness_score = max(0, 100 - abs(face_brightness - 150) * 2)
    contrast_score = min(100, face_contrast * 3)
    face_lighting_score = int((brightness_score + contrast_score) / 2)
    
    return {
        "face_detection": 100,  # Face was detected
        "face_positioning": int(main_face['centering_score']),
        "face_size": int(main_face['size_score']),
        "face_lighting": face_lighting_score
    }

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

def analyze_eye_contact(image_array: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze if the person is looking at the camera."""
    if not faces:
        return {"eye_contact": False, "eye_contact_score": 0, "gaze_direction": "unknown"}
    
    # Use the largest face
    main_face = max(faces, key=lambda f: f['face_ratio'])
    x, y, w, h = main_face['bbox']
    
    # Extract face region
    face_region = image_array[y:y+h, x:x+w]
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # Load eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    if len(eyes) < 2:
        return {"eye_contact": False, "eye_contact_score": 30, "gaze_direction": "unclear"}
    
    # Analyze eye symmetry and positioning (basic gaze estimation)
    eye1, eye2 = sorted(eyes, key=lambda e: e[0])[:2]  # Left and right eyes
    
    # Calculate eye centers
    eye1_center = (eye1[0] + eye1[2]//2, eye1[1] + eye1[3]//2)
    eye2_center = (eye2[0] + eye2[2]//2, eye2[1] + eye2[3]//2)
    
    # Check eye symmetry (indicates looking at camera)
    eye_distance = abs(eye2_center[0] - eye1_center[0])
    eye_height_diff = abs(eye2_center[1] - eye1_center[1])
    
    # Face center
    face_center_x = w // 2
    
    # Estimate gaze direction based on eye positioning
    avg_eye_x = (eye1_center[0] + eye2_center[0]) // 2
    
    # Calculate eye contact score
    symmetry_score = max(0, 100 - (eye_height_diff * 10))
    center_score = max(0, 100 - abs(avg_eye_x - face_center_x) * 5)
    
    eye_contact_score = int((symmetry_score + center_score) / 2)
    
    # Determine gaze direction
    if abs(avg_eye_x - face_center_x) < w * 0.1:
        gaze_direction = "direct"
        eye_contact = True
    elif avg_eye_x < face_center_x - w * 0.1:
        gaze_direction = "left"
        eye_contact = False
    elif avg_eye_x > face_center_x + w * 0.1:
        gaze_direction = "right" 
        eye_contact = False
    else:
        gaze_direction = "slightly_off"
        eye_contact = False
    
    return {
        "eye_contact": eye_contact,
        "eye_contact_score": max(20, eye_contact_score),
        "gaze_direction": gaze_direction
    }

def analyze_color_harmony(image_array: np.ndarray) -> Dict[str, Any]:
    """Analyze color balance, temperature, and harmony."""
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    
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

def analyze_time_of_day(image_array: np.ndarray) -> Dict[str, Any]:
    """Identify lighting conditions and time of day characteristics."""
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    
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

def generate_suggestions(scores: Dict[str, int], has_face: bool = False) -> List[str]:
    """Generate intelligent, personalized suggestions based on analysis scores."""
    suggestions = []
    priority_suggestions = []  # High priority issues
    improvement_tips = []      # General improvement tips
    encouragement = []         # Positive feedback
    
    # Calculate overall score for context
    overall_score = sum(scores.values()) / len(scores)
    
    # === FACE-SPECIFIC ANALYSIS ===
    if has_face:
        face_pos = scores.get("face_positioning", 0)
        face_size = scores.get("face_size", 0) 
        face_light = scores.get("face_lighting", 0)
        
        # Face positioning suggestions
        if face_pos < 30:
            priority_suggestions.append("🎯 URGENT: Center your face in the frame - you're significantly off-center")
        elif face_pos < 60:
            improvement_tips.append("📐 Face positioning: Move slightly to center yourself better in the frame")
        elif face_pos < 85:
            improvement_tips.append("✨ Almost perfect positioning! Minor centering adjustment would be ideal")
        else:
            encouragement.append("🎯 Perfect face positioning! You're centered beautifully")
        
        # Face size suggestions
        if face_size < 20:
            priority_suggestions.append("🔍 CRITICAL: Face is too small - move much closer to the camera")
        elif face_size < 40:
            improvement_tips.append("📏 Face size: Move closer to the camera for better framing")
        elif face_size < 60:
            improvement_tips.append("📐 Face size is okay, but moving slightly closer would improve the shot")
        elif face_size < 85:
            improvement_tips.append("👌 Good face size! Maybe just a touch closer for optimal framing")
        else:
            encouragement.append("📏 Perfect face size! Great framing distance")
        
        # Face lighting suggestions
        if face_light < 25:
            priority_suggestions.append("💡 URGENT: Face is too dark - face a window or add lighting")
        elif face_light < 50:
            improvement_tips.append("🌟 Face lighting: Turn toward a light source or move to brighter area")
        elif face_light < 75:
            improvement_tips.append("💡 Face lighting is decent but could be more even")
        else:
            encouragement.append("🌟 Beautiful face lighting! Very well lit")
    else:
        improvement_tips.append("👤 No face detected - ensure your face is visible and well-lit for selfie analysis")
    
    # === TECHNICAL QUALITY ANALYSIS ===
    lighting = scores.get("lighting", 0)
    sharpness = scores.get("sharpness", 0)
    contrast = scores.get("contrast", 0)
    background = scores.get("background_clutter", 0)
    
    # === ADVANCED ANALYSIS ===
    emotion_quality = scores.get("emotion_quality", 0)
    eye_contact = scores.get("eye_contact", 0)
    color_harmony = scores.get("color_harmony", 0)
    composition = scores.get("composition", 0)
    naturalness = scores.get("naturalness", 0)
    style = scores.get("style", 0)
    lighting_time = scores.get("lighting_time", 0)
    
    # Lighting suggestions
    if lighting < 25:
        priority_suggestions.append("💡 CRITICAL: Very poor lighting - move to natural light or add illumination")
    elif lighting < 50:
        improvement_tips.append("🌤️ Lighting needs work: Try window light, golden hour, or soft indoor lighting")
    elif lighting < 75:
        improvement_tips.append("☀️ Good lighting foundation - minor adjustments could make it excellent")
    elif lighting < 90:
        encouragement.append("🌟 Great lighting! Just a touch more balance would be perfect")
    else:
        encouragement.append("💡 Exceptional lighting! Professional quality")
    
    # Sharpness suggestions  
    if sharpness < 20:
        priority_suggestions.append("📱 CRITICAL: Very blurry - use timer, tripod, or steady your hands")
    elif sharpness < 45:
        improvement_tips.append("🔍 Sharpness issue: Hold phone steadier, tap to focus, or use burst mode")
    elif sharpness < 70:
        improvement_tips.append("📸 Slight blur detected - ensure focus lock before shooting")
    elif sharpness < 85:
        encouragement.append("🎯 Good sharpness! Very crisp image")
    else:
        encouragement.append("🔍 Tack sharp! Excellent focus quality")
    
    # Contrast suggestions
    if contrast < 30:
        improvement_tips.append("⚡ Low contrast: Try varied lighting or adjust camera exposure")
    elif contrast < 60:
        improvement_tips.append("🎨 Contrast could be enhanced - experiment with lighting angles")
    elif contrast < 80:
        encouragement.append("⚡ Nice contrast levels!")
    else:
        encouragement.append("🎨 Excellent contrast! Great dynamic range")
    
    # Background suggestions
    if background < 35:
        improvement_tips.append("🎨 Very busy background - try a plain wall, outdoors, or portrait mode")
    elif background < 55:
        improvement_tips.append("🖼️ Background is a bit cluttered - simplify or use depth of field")
    elif background < 75:
        improvement_tips.append("🎭 Background is decent but could be cleaner for better subject focus")
    elif background < 85:
        encouragement.append("🖼️ Nice clean background!")
    else:
        encouragement.append("🎨 Perfect background! Excellent subject isolation")
    
    # === ADVANCED FEATURE SUGGESTIONS ===
    # Emotion quality suggestions
    if has_face and emotion_quality < 40:
        improvement_tips.append("😊 Try a more natural expression - smile slightly or look confident")
    elif has_face and emotion_quality > 80:
        encouragement.append("😊 Great expression! Very engaging and natural")
    
    # Eye contact suggestions
    if has_face and eye_contact < 50:
        improvement_tips.append("👁️ Look directly at the camera lens for better connection")
    elif has_face and eye_contact > 75:
        encouragement.append("👁️ Perfect eye contact! Very engaging")
    
    # Color harmony suggestions
    if color_harmony < 40:
        improvement_tips.append("🎨 Color balance could be improved - try adjusting white balance or lighting")
    elif color_harmony > 80:
        encouragement.append("🎨 Beautiful color harmony! Excellent visual appeal")
    
    # Composition suggestions
    if composition < 40:
        improvement_tips.append("📐 Try positioning yourself off-center using the rule of thirds")
    elif composition > 80:
        encouragement.append("📐 Excellent composition! Great use of visual principles")
    
    # Naturalness suggestions
    if naturalness < 50:
        improvement_tips.append("🌟 Photo appears heavily filtered - try more natural processing")
    elif naturalness > 85:
        encouragement.append("🌟 Beautiful natural look! No over-processing detected")
    
    # Style suggestions
    if style < 50:
        improvement_tips.append("👕 Consider outfit coordination and simpler patterns for better photos")
    elif style > 80:
        encouragement.append("👕 Excellent style choices! Great color coordination")
    
    # Lighting/Time suggestions
    if lighting_time < 40:
        improvement_tips.append("🌅 Try shooting during golden hour or with better lighting conditions")
    elif lighting_time > 85:
        encouragement.append("🌅 Perfect lighting conditions! Optimal time and setup")
    
    # === OVERALL ASSESSMENT & MOTIVATION ===
    if overall_score >= 85:
        encouragement.insert(0, "🏆 OUTSTANDING! This is professional-quality work!")
    elif overall_score >= 75:
        encouragement.insert(0, "🌟 EXCELLENT photo! You have great photography instincts")
    elif overall_score >= 65:
        encouragement.insert(0, "👍 GOOD quality! You're on the right track")
    elif overall_score >= 50:
        improvement_tips.insert(0, "📈 DECENT foundation - focus on the key areas below for big improvements")
    elif overall_score >= 35:
        improvement_tips.insert(0, "🔧 NEEDS WORK - tackle the priority issues first for quick wins")
    else:
        priority_suggestions.insert(0, "🚨 MULTIPLE ISSUES - start with lighting and stability fundamentals")
    
    # === COMBINE SUGGESTIONS IN ORDER OF IMPORTANCE ===
    # Priority issues first (max 2 to avoid overwhelming)
    suggestions.extend(priority_suggestions[:2])
    
    # Then improvement tips (max 3)
    suggestions.extend(improvement_tips[:3])
    
    # Finally encouragement (max 2)
    suggestions.extend(encouragement[:2])
    
    # === BONUS TIPS BASED ON SCORE PATTERNS ===
    face_light = scores.get("face_lighting", 0) if has_face else 0
    if lighting > 80 and sharpness > 80 and background > 80 and face_light > 80:
        suggestions.append("🎓 PRO TIP: Your technical skills are excellent! Try creative angles or expressions")
    elif lighting < 40 and sharpness < 40:
        suggestions.append("💡 QUICK WIN: Fix lighting first - it will automatically improve apparent sharpness")
    elif has_face and face_size > 80 and face_pos > 80:
        suggestions.append("📸 GREAT JOB: Your composition skills are strong! Focus on lighting next")
    
    return suggestions if suggestions else ["✅ Analysis complete - all metrics look good!"]

def score_selfie(image: Image.Image) -> str:
    if image is None:
        return json.dumps({
            "scores": {},
            "suggestions": ["Please upload an image to get a rating."]
        }, indent=2)

    # Convert PIL Image to numpy array for OpenCV
    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)
    
    # Detect faces first
    faces = detect_faces(img_array)
    has_face = len(faces) > 0
    
    # Run all analysis functions
    lighting_score = analyze_lighting(img_array)
    sharpness_score = analyze_sharpness(img_array)
    contrast_score = analyze_contrast(img_array)
    background_score = analyze_background_complexity(img_array)
    
    # Base scores
    scores = {
        "lighting": lighting_score,
        "sharpness": sharpness_score,
        "contrast": contrast_score,
        "background_clutter": background_score,
    }
    
    # Add face-specific scores if face detected
    if has_face:
        face_scores = analyze_face_quality(img_array, faces)
        scores.update(face_scores)
        
        # Add advanced face analysis
        emotion_data = analyze_emotion(img_array, faces)
        eye_contact_data = analyze_eye_contact(img_array, faces)
        
        scores["emotion_quality"] = emotion_data["emotion_score"]
        scores["eye_contact"] = eye_contact_data["eye_contact_score"]
    
    # Add advanced analysis for all photos
    color_data = analyze_color_harmony(img_array)
    composition_data = analyze_rule_of_thirds(img_array, faces)
    filter_data = analyze_beauty_filter(img_array, faces)
    style_data = analyze_outfit_style(img_array, faces)
    lighting_time_data = analyze_time_of_day(img_array)
    
    scores["color_harmony"] = color_data["color_harmony_score"]
    scores["composition"] = composition_data["rule_of_thirds_score"]
    scores["naturalness"] = filter_data["naturalness_score"]
    scores["style"] = style_data["style_score"]
    scores["lighting_time"] = lighting_time_data["lighting_quality_score"]
    
    # Generate intelligent suggestions
    suggestions = generate_suggestions(scores, has_face)
    
    # Calculate overall quality score
    overall_score = sum(scores.values()) / len(scores)
    
    # Add face detection info
    face_info = {
        "faces_detected": len(faces),
        "has_face": has_face
    }
    
    if has_face:
        main_face = max(faces, key=lambda f: f['face_ratio'])
        face_info["main_face_size_percent"] = round(main_face['face_ratio'] * 100, 1)
    
    # Collect all detailed analysis data
    advanced_analysis = {
        "color_analysis": color_data,
        "composition_analysis": composition_data,
        "filter_analysis": filter_data,
        "style_analysis": style_data,
        "lighting_time_analysis": lighting_time_data
    }
    
    if has_face:
        advanced_analysis["emotion_analysis"] = emotion_data
        advanced_analysis["eye_contact_analysis"] = eye_contact_data

    payload: Dict[str, Any] = {
        "scores": scores,
        "overall_quality": round(overall_score, 1),
        "face_info": face_info,
        "advanced_analysis": advanced_analysis,
        "suggestions": suggestions,
        "analysis_note": "Advanced computer vision with face detection, emotion analysis, and composition assessment.",
    }

    return json.dumps(payload, indent=2)


def analyze_realtime(image: Image.Image) -> Dict[str, Any]:
    """Optimized real-time analysis for live camera feed."""
    if image is None:
        return {
            "status": "no_image",
            "quick_tips": ["Position yourself in front of the camera"],
            "scores": {}
        }
    
    # Convert and resize for faster processing
    img_rgb = image.convert("RGB")
    # Resize to smaller size for faster real-time processing
    img_rgb.thumbnail((640, 480), Image.Resampling.LANCZOS)
    img_array = np.array(img_rgb)
    
    # Quick face detection
    faces = detect_faces(img_array)
    has_face = len(faces) > 0
    
    # Fast analysis - only essential metrics for real-time
    quick_scores = {}
    tips = []
    
    if has_face:
        # Quick face analysis
        face_scores = analyze_face_quality(img_array, faces)
        emotion_data = analyze_emotion(img_array, faces)
        eye_data = analyze_eye_contact(img_array, faces)
        
        quick_scores = {
            "face_detected": True,
            "face_positioning": face_scores.get("face_positioning", 0),
            "face_lighting": face_scores.get("face_lighting", 0),
            "emotion_quality": emotion_data.get("emotion_score", 0),
            "eye_contact": eye_data.get("eye_contact_score", 0)
        }
        
        # Real-time tips based on scores
        if face_scores.get("face_positioning", 0) < 60:
            tips.append("🎯 Center yourself in the frame")
        if face_scores.get("face_lighting", 0) < 50:
            tips.append("💡 Face more light or move to brighter area")
        if emotion_data.get("emotion_score", 0) < 60:
            tips.append("😊 Try a natural smile or confident expression")
        if eye_data.get("eye_contact_score", 0) < 60:
            tips.append("👁️ Look directly at the camera")
            
    else:
        quick_scores = {"face_detected": False}
        tips = ["👤 Move into frame - no face detected"]
    
    # Quick technical analysis
    lighting_score = analyze_lighting(img_array)
    sharpness_score = analyze_sharpness(img_array)
    
    quick_scores.update({
        "lighting": lighting_score,
        "sharpness": sharpness_score
    })
    
    # Technical tips
    if lighting_score < 50:
        tips.append("💡 Move to better lighting")
    if sharpness_score < 40:
        tips.append("📱 Hold camera steady")
    
    # Overall quality for real-time
    overall = sum([v for v in quick_scores.values() if isinstance(v, (int, float))]) / max(1, len([v for v in quick_scores.values() if isinstance(v, (int, float))]))
    
    return {
        "status": "analyzing",
        "scores": quick_scores,
        "overall_score": round(overall, 1),
        "quick_tips": tips[:4],  # Limit to 4 tips for real-time
        "has_face": has_face
    }

def create_realtime_feedback_html(analysis: Dict[str, Any]) -> str:
    """Create real-time feedback overlay HTML."""
    if analysis["status"] == "no_image":
        return """
        <div style="text-align: center; padding: 20px; background: rgba(0,0,0,0.8); color: white; border-radius: 10px;">
            <h3>📸 Position yourself for analysis</h3>
        </div>
        """
    
    scores = analysis.get("scores", {})
    tips = analysis.get("quick_tips", [])
    overall = analysis.get("overall_score", 0)
    has_face = analysis.get("has_face", False)
    
    # Color coding for real-time feedback
    if overall >= 75:
        status_color = "#22c55e"
        status_text = "🎉 Excellent!"
    elif overall >= 60:
        status_color = "#3b82f6"
        status_text = "👍 Good"
    elif overall >= 40:
        status_color = "#f59e0b"
        status_text = "📈 Getting Better"
    else:
        status_color = "#ef4444"
        status_text = "🔧 Needs Work"
    
    # Face detection indicator
    face_indicator = "✅ Face Detected" if has_face else "❌ No Face"
    face_color = "#22c55e" if has_face else "#ef4444"
    
    # Quick tips HTML
    tips_html = ""
    for tip in tips:
        tips_html += f'<div style="margin: 5px 0; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 5px;">{tip}</div>'
    
    return f"""
    <div style="background: rgba(0,0,0,0.85); color: white; padding: 20px; border-radius: 15px; backdrop-filter: blur(5px); font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <!-- Overall Status -->
        <div style="text-align: center; margin-bottom: 15px;">
            <div style="font-size: 1.5em; color: {status_color}; font-weight: bold;">{overall:.1f}/100</div>
            <div style="color: {status_color}; font-weight: 600;">{status_text}</div>
        </div>
        
        <!-- Face Detection -->
        <div style="text-align: center; margin-bottom: 15px; color: {face_color}; font-weight: 600;">
            {face_indicator}
        </div>
        
        <!-- Quick Tips -->
        <div style="margin-top: 15px;">
            <div style="font-weight: 600; margin-bottom: 10px; text-align: center;">💡 Live Tips:</div>
            {tips_html if tips else '<div style="text-align: center; color: #22c55e;">Perfect! Ready to capture 📸</div>'}
        </div>
    </div>
    """


def create_csv_report(data: Dict[str, Any]) -> str:
    """Create a CSV report from analysis data."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['SQRT Analysis Report'])
    writer.writerow(['Generated:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])
    
    # Overall score
    writer.writerow(['Overall Quality Score', f"{data['overall_quality']}/100"])
    writer.writerow([])
    
    # Face detection info
    if data.get('face_info'):
        face_info = data['face_info']
        writer.writerow(['Face Detection Results'])
        writer.writerow(['Faces Detected', face_info.get('faces_detected', 0)])
        writer.writerow(['Face Found', 'Yes' if face_info.get('has_face', False) else 'No'])
        if face_info.get('has_face', False):
            writer.writerow(['Face Size %', f"{face_info.get('main_face_size_percent', 0)}%"])
        writer.writerow([])
    
    # Individual scores
    writer.writerow(['Detailed Scores'])
    writer.writerow(['Metric', 'Score (/100)', 'Rating'])
    
    for metric, score in data['scores'].items():
        rating = 'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Fair' if score >= 40 else 'Poor'
        metric_name = metric.replace('_', ' ').title()
        writer.writerow([metric_name, score, rating])
    
    writer.writerow([])
    
    # Suggestions
    writer.writerow(['Suggestions for Improvement'])
    for i, suggestion in enumerate(data['suggestions'], 1):
        writer.writerow([f'Tip {i}', suggestion])
    
    writer.writerow([])
    writer.writerow(['Note', data.get('analysis_note', '')])
    
    return output.getvalue()

def create_text_report(data: Dict[str, Any]) -> str:
    """Create a detailed text report from analysis data."""
    report = []
    report.append("="*60)
    report.append("SQRT - SELFIE QUALITY RATER ANALYSIS REPORT")
    report.append("="*60)
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall score with visual bar
    overall = data['overall_quality']
    bar_length = 50
    filled_length = int(overall / 100 * bar_length)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    report.append(f"📊 OVERALL QUALITY SCORE: {overall:.1f}/100")
    report.append(f"   [{bar}]")
    
    if overall >= 80:
        report.append("   🏆 EXCELLENT QUALITY!")
    elif overall >= 60:
        report.append("   👍 GOOD QUALITY")
    elif overall >= 40:
        report.append("   📈 NEEDS IMPROVEMENT")
    else:
        report.append("   🔧 POOR QUALITY")
    
    report.append("")
    
    # Face detection info
    if data.get('face_info'):
        face_info = data['face_info']
        report.append("👤 FACE DETECTION RESULTS")
        report.append("-" * 30)
        report.append(f"Faces detected: {face_info.get('faces_detected', 0)}")
        report.append(f"Face found: {'✓ Yes' if face_info.get('has_face', False) else '✗ No'}")
        if face_info.get('has_face', False):
            report.append(f"Face size: {face_info.get('main_face_size_percent', 0):.1f}% of image")
        report.append("")
    
    # Detailed scores
    report.append("📈 DETAILED ANALYSIS SCORES")
    report.append("-" * 30)
    
    for metric, score in data['scores'].items():
        metric_name = metric.replace('_', ' ').title()
        
        # Create mini progress bar
        mini_bar_length = 20
        mini_filled = int(score / 100 * mini_bar_length)
        mini_bar = "█" * mini_filled + "░" * (mini_bar_length - mini_filled)
        
        rating = "🟢" if score >= 80 else "🟡" if score >= 60 else "🟠" if score >= 40 else "🔴"
        
        report.append(f"{metric_name:20} {score:3d}/100 [{mini_bar}] {rating}")
    
    report.append("")
    
    # Suggestions
    report.append("💡 SUGGESTIONS FOR IMPROVEMENT")
    report.append("-" * 30)
    for i, suggestion in enumerate(data['suggestions'], 1):
        report.append(f"{i:2d}. {suggestion}")
    
    report.append("")
    report.append("=" * 60)
    report.append("📝 NOTE")
    report.append(data.get('analysis_note', ''))
    report.append("=" * 60)
    
    return "\n".join(report)

def save_analysis_results(image: Image.Image, analysis_data: Dict[str, Any]) -> Tuple[str, str, str]:
    """Save analysis results and return file paths for download."""
    if image is None or not analysis_data:
        return None, None, None
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create JSON file
    json_content = json.dumps(analysis_data, indent=2)
    json_path = f"sqrt_analysis_{timestamp}.json"
    
    # Create CSV file
    csv_content = create_csv_report(analysis_data)
    csv_path = f"sqrt_analysis_{timestamp}.csv"
    
    # Create text report
    text_content = create_text_report(analysis_data)
    text_path = f"sqrt_report_{timestamp}.txt"
    
    # Write files
    with open(json_path, 'w') as f:
        f.write(json_content)
    
    with open(csv_path, 'w') as f:
        f.write(csv_content)
        
    with open(text_path, 'w') as f:
        f.write(text_content)
    
    return json_path, csv_path, text_path

def create_score_html(scores: Dict[str, int], overall_score: float, suggestions: List[str], face_info: Dict[str, Any] = None) -> str:
    """Create beautiful grouped HTML visualization of scores with categorized analytics."""
    
    def get_color(score: int) -> str:
        if score >= 80: return "#22c55e"  # green
        elif score >= 60: return "#eab308"  # yellow
        elif score >= 40: return "#f97316"  # orange
        else: return "#ef4444"  # red
    
    def get_overall_color(score: float) -> str:
        if score >= 80: return "#22c55e"
        elif score >= 60: return "#3b82f6"  # blue
        elif score >= 40: return "#f97316"
        else: return "#ef4444"
    
    def create_score_bar(metric: str, score: int, icon: str = "📊") -> str:
        color = get_color(score)
        metric_name = metric.replace("_", " ").title()
        
        return f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: 600; color: #374151;">{icon} {metric_name}</span>
                <span style="font-weight: 700; color: {color}; font-size: 0.9em;">{score}/100</span>
            </div>
            <div style="background-color: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
                <div style="background-color: {color}; height: 100%; width: {score}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    # Categorize metrics
    basic_metrics = {
        "lighting": "💡",
        "sharpness": "🔍", 
        "contrast": "⚡",
        "background_clutter": "🖼️"
    }
    
    face_metrics = {
        "face_positioning": "🎯",
        "face_size": "📏", 
        "face_lighting": "🌟",
        "emotion_quality": "😊",
        "eye_contact": "👁️"
    }
    
    advanced_metrics = {
        "color_harmony": "🎨",
        "composition": "📐",
        "naturalness": "🌟",
        "style": "👕",
        "lighting_time": "🌅"
    }
    
    # Create categorized score sections
    def create_category_section(title: str, metrics: dict, icon: str) -> str:
        section_html = f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 15px;">
            <h3 style="margin: 0 0 15px 0; color: #1f2937; font-size: 1.1em; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px;">
                {icon} {title}
            </h3>
        """
        
        for metric, metric_icon in metrics.items():
            if metric in scores:
                section_html += create_score_bar(metric, scores[metric], metric_icon)
        
        section_html += "</div>"
        return section_html
    
    # Build sections
    basic_section = create_category_section("Technical Quality", basic_metrics, "⚙️")
    
    face_section = ""
    if face_info and face_info.get("has_face", False):
        face_section = create_category_section("Face Analysis", face_metrics, "👤")
    
    advanced_section = create_category_section("Advanced Analysis", advanced_metrics, "🧠")
    
    # Create suggestions HTML
    suggestions_html = ""
    for suggestion in suggestions:
        suggestions_html += f'<li style="margin-bottom: 8px; color: #4b5563;">{suggestion}</li>'
    
    # Overall score badge
    overall_color = get_overall_color(overall_score)
    
    # Face detection info
    face_badge = ""
    if face_info:
        if face_info.get("has_face", False):
            face_count = face_info.get("faces_detected", 0)
            face_size = face_info.get("main_face_size_percent", 0)
            face_badge = f"""
            <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #22c55e15, #22c55e05); border-radius: 10px; border: 1px solid #22c55e30;">
                <div style="font-size: 1.2em; color: #22c55e; font-weight: 600;">
                    👤 Face Detected! ({face_count} face{'s' if face_count != 1 else ''})
                </div>
                <div style="font-size: 0.9em; color: #6b7280; margin-top: 5px;">
                    Face size: {face_size}% of image
                </div>
            </div>
            """
        else:
            face_badge = f"""
            <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #f97316015, #f9731605); border-radius: 10px; border: 1px solid #f9731630;">
                <div style="font-size: 1.2em; color: #f97316; font-weight: 600;">
                    😵 No Face Detected
                </div>
                <div style="font-size: 0.9em; color: #6b7280; margin-top: 5px;">
                    Make sure your face is visible and well-lit
                </div>
            </div>
            """
    
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 700px; margin: 0 auto;">
        {face_badge}
        
        <!-- Overall Score -->
        <div style="text-align: center; margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, {overall_color}15, {overall_color}05); border-radius: 15px; border: 1px solid {overall_color}30;">
            <h2 style="margin: 0 0 10px 0; color: #1f2937;">Overall Quality Score</h2>
            <div style="font-size: 3em; font-weight: 800; color: {overall_color}; margin: 10px 0;">
                {overall_score:.1f}/100
            </div>
            <div style="font-size: 1.1em; color: #6b7280;">
                {"🏆 Excellent!" if overall_score >= 80 else "👍 Good!" if overall_score >= 60 else "📈 Needs Improvement" if overall_score >= 40 else "🔧 Poor Quality"}
            </div>
        </div>
        
        <!-- Categorized Analysis Sections -->
        {basic_section}
        {face_section}
        {advanced_section}
        
        <!-- Suggestions -->
        <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="margin: 0 0 15px 0; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">💡 Personalized Suggestions</h3>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                {suggestions_html}
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 20px; padding: 15px; background: #f8fafc; border-radius: 10px;">
            <small style="color: #6b7280; font-style: italic;">
                ✨ Advanced AI analysis with 7 categories • Rating photo quality, not the person
            </small>
        </div>
    </div>
    """
    
    return html

def analyze_selfie_visual(image: Image.Image) -> str:
    """Analyze selfie and return beautiful HTML visualization."""
    if image is None:
        return """
        <div style="text-align: center; padding: 40px; color: #6b7280;">
            <h3>📸 Upload a selfie to get started!</h3>
            <p>Your photo will be analyzed for lighting, sharpness, contrast, background quality, and face detection.</p>
        </div>
        """
    
    # Get the JSON analysis
    json_result = score_selfie(image)
    data = json.loads(json_result)
    
    # Create beautiful HTML visualization
    return create_score_html(
        data["scores"], 
        data["overall_quality"], 
        data["suggestions"],
        data.get("face_info")
    )

def analyze_and_generate_downloads(image: Image.Image) -> Tuple[str, str, str, str, str]:
    """Analyze image and generate both HTML and download files."""
    if image is None:
        empty_html = """
        <div style="text-align: center; padding: 40px; color: #6b7280;">
            <h3>📸 Upload a selfie to get started!</h3>
            <p>Your photo will be analyzed for lighting, sharpness, contrast, background quality, and face detection.</p>
        </div>
        """
        return empty_html, None, None, None, "🔼 Upload and analyze a photo to enable downloads"
    
    # Get the JSON analysis
    json_result = score_selfie(image)
    data = json.loads(json_result)
    
    # Create beautiful HTML visualization
    html_result = create_score_html(
        data["scores"], 
        data["overall_quality"], 
        data["suggestions"],
        data.get("face_info")
    )
    
    # Generate download files
    json_path, csv_path, txt_path = save_analysis_results(image, data)
    
    download_message = f"✅ Analysis complete! Download your reports above (Generated at {datetime.datetime.now().strftime('%H:%M:%S')})"
    
    return html_result, json_path, csv_path, txt_path, download_message

def main() -> None:
    with gr.Blocks(
        title="Selfie Quality Rater (SQRT)",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        css="""
        .gradio-container {
            max-width: 1000px !important;
            margin: auto !important;
        }
        .upload-container {
            border: 2px dashed #d1d5db !important;
            border-radius: 12px !important;
        }
        .upload-container:hover {
            border-color: #3b82f6 !important;
        }
        .mode-tab {
            font-size: 1.1em !important;
            font-weight: 600 !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📸 Selfie Quality Rater (SQRT) ✨
        
        **Professional AI-powered selfie analysis with real-time feedback!**
        
        Choose your mode: Upload photos for detailed analysis or use live camera for real-time coaching.
        """)
        
        # Mode Selection Tabs
        with gr.Tabs() as mode_tabs:
            
            # Upload Mode Tab
            with gr.TabItem("📤 Upload Mode", elem_classes=["mode-tab"]) as upload_tab:
                gr.Markdown("### 📋 Comprehensive Analysis")
                gr.Markdown("Upload a photo for detailed analysis across all 7 categories with downloadable reports.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil", 
                            label="📤 Upload Your Selfie",
                            elem_classes=["upload-container"]
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔍 Analyze Photo", 
                                variant="primary", 
                                size="lg"
                            )
                        
                        gr.Markdown("### 💾 Download Reports")
                        
                        with gr.Row():
                            download_json = gr.File(
                                label="📄 JSON Data",
                                visible=False
                            )
                            download_csv = gr.File(
                                label="📊 CSV Report", 
                                visible=False
                            )
                            download_txt = gr.File(
                                label="📝 Text Report",
                                visible=False
                            )
                        
                        download_info = gr.Markdown(
                            "🔼 Upload and analyze a photo to enable downloads",
                            visible=True
                        )
                        
                    with gr.Column(scale=1):
                        output_html = gr.HTML(
                            label="📊 Analysis Results",
                            value="""
                            <div style="text-align: center; padding: 40px; color: #6b7280;">
                                <h3>👈 Upload a photo to see results here!</h3>
                            </div>
                            """
                        )
            
            # Real-time Mode Tab
            with gr.TabItem("📹 Live Mode", elem_classes=["mode-tab"]) as realtime_tab:
                gr.Markdown("### 📸 Instant Webcam Analysis")
                gr.Markdown("Take a selfie with your webcam and get immediate feedback!")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Webcam input that will capture images
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            type="pil",
                            label="📷 Take Your Selfie",
                            mirror_webcam=False
                        )
                        
                        # Auto-analyze when webcam captures a photo
                        gr.Markdown("📝 **Instructions:**")
                        gr.Markdown("1. 📷 Click the camera button above to take a selfie")
                        gr.Markdown("2. 🎯 Your photo will be analyzed automatically")
                        gr.Markdown("3. 📊 Results will appear on the right")
                        
                    with gr.Column(scale=1):
                        webcam_feedback = gr.HTML(
                            label="📊 Instant Analysis Results",
                            value="""
                            <div style="text-align: center; padding: 40px; color: #6b7280;">
                                <h3>📷 Take a selfie to see instant analysis!</h3>
                                <p>Results will appear here automatically</p>
                            </div>
                            """
                        )
        
        # === UPLOAD MODE EVENT HANDLERS ===
        
        # Auto-analyze when image is uploaded and generate downloads
        image_input.change(
            fn=analyze_and_generate_downloads,
            inputs=image_input,
            outputs=[output_html, download_json, download_csv, download_txt, download_info]
        )
        
        # Manual analyze button
        analyze_btn.click(
            fn=analyze_and_generate_downloads,
            inputs=image_input,
            outputs=[output_html, download_json, download_csv, download_txt, download_info]
        )
        
        # Show/hide download files based on analysis
        def update_download_visibility(json_file, csv_file, txt_file):
            has_files = json_file is not None
            return [
                gr.File(visible=has_files),  # JSON
                gr.File(visible=has_files),  # CSV
                gr.File(visible=has_files),  # TXT
            ]
        
        download_json.change(
            fn=update_download_visibility,
            inputs=[download_json, download_csv, download_txt],
            outputs=[download_json, download_csv, download_txt]
        )
        
        # === WEBCAM MODE EVENT HANDLERS ===
        
        # Auto-analyze webcam captures
        def analyze_webcam_selfie(image):
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            print(f"\n📷 [{timestamp}] WEBCAM SELFIE AUTO-ANALYSIS")
            print(f"   ├─ Image received: {image is not None}")
            if image is not None:
                print(f"   ├─ Image type: {type(image)}")
                print(f"   └─ Image size: {image.size if hasattr(image, 'size') else 'unknown'}")
            
            if image is None:
                print(f"   🔴 NO IMAGE - Waiting for selfie...")
                return """
                <div style="text-align: center; padding: 40px; color: #6b7280;">
                    <h3>📷 Take a selfie to see instant analysis!</h3>
                    <p>Results will appear here automatically</p>
                </div>
                """
            
            try:
                print(f"   🟢 AUTO-ANALYZING WEBCAM SELFIE...")
                # Use the same detailed analysis as upload mode
                analysis_result = analyze_selfie_visual(image)
                print(f"   ✅ Webcam selfie analysis complete!")
                return analysis_result
            except Exception as e:
                print(f"   🔴 ERROR during webcam analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                return f"""
                <div style="text-align: center; padding: 40px; color: #ef4444; background: rgba(239, 68, 68, 0.1); border-radius: 15px;">
                    <h3>❌ Analysis Error</h3>
                    <p>Error: {str(e)}</p>
                    <p>Try taking another selfie!</p>
                </div>
                """
        
        # Auto-analyze when webcam captures a photo
        print("🔧 Setting up webcam_input.change event handler...")
        webcam_input.change(
            fn=analyze_webcam_selfie,
            inputs=webcam_input,
            outputs=webcam_feedback,
            show_progress="full"
        )
        print("✅ Webcam event handler set up successfully!")
        
        gr.Markdown("""
        ---
        ### 🔬 What We Analyze:
        - **Face Detection**: Automatic face recognition and positioning
        - **Face Quality**: Face-specific lighting, size, and centering
        - **Lighting**: Exposure balance and illumination quality
        - **Sharpness**: Focus clarity using edge detection
        - **Contrast**: Dynamic range and depth
        - **Background**: Clutter and complexity analysis
        
        ### 💾 Download Options:
        - **JSON**: Raw analysis data for developers
        - **CSV**: Spreadsheet-compatible data for analysis
        - **TXT**: Detailed text report with visual progress bars
        
        *Built with advanced computer vision & face detection • Privacy-first: photos are processed locally*
        """)
    
    demo.launch(share=True, show_api=False, show_error=True)


if __name__ == "__main__":
    main()
