"""
Main analysis orchestrator that coordinates all analysis modules.
"""

import json
import numpy as np
from PIL import Image
from typing import Dict, Any

from .face_detection import detect_faces, analyze_face_quality, analyze_eye_contact
from .image_analysis import (
    analyze_sharpness, analyze_lighting, analyze_contrast, 
    analyze_background_complexity, analyze_color_harmony, analyze_time_of_day
)
from .advanced_analysis import (
    analyze_emotion, analyze_rule_of_thirds, analyze_beauty_filter, analyze_outfit_style
)
from .suggestion_generator import generate_suggestions


def score_selfie(image: Image.Image) -> str:
    """Main function to score a selfie image and return JSON results."""
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
