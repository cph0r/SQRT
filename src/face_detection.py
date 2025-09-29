"""
Face detection and face-specific analysis module.
"""

import cv2
import numpy as np
from typing import Dict, Any, List


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
