"""
Intelligent suggestion generation module based on analysis scores.
"""

from typing import Dict, List


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
