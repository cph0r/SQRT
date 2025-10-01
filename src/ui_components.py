"""
Gradio UI components and interface definitions.
Optimized for production with proper logging and error handling.
"""

import gradio as gr
import datetime
import json
import time
import logging
from PIL import Image
from typing import Tuple

from .analyzer import score_selfie, analyze_realtime
from .report_generator import (
    create_score_html, 
    create_realtime_feedback_html, 
    save_analysis_results
)

# Configure logging
logger = logging.getLogger(__name__)

# Global variable for throttling real-time analysis
_last_analysis_time = 0
_analysis_interval = 2.0  # seconds


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


def analyze_webcam_realtime(image: Image.Image) -> str:
    """Continuous real-time analysis for live webcam feed."""
    global _last_analysis_time
    
    current_time = time.time()
    
    if image is None:
        return """
        <div style="text-align: center; padding: 40px; color: #6b7280;">
            <h3>📷 Waiting for webcam feed...</h3>
            <p>Live analysis will appear automatically</p>
        </div>
        """
    
    # Throttle analysis to avoid overwhelming the system
    time_since_last = current_time - _last_analysis_time
    if time_since_last < _analysis_interval and _last_analysis_time > 0:
        # Return None to keep previous result displayed
        return gr.update()  # No update, keep previous output
    
    try:
        # Update last analysis time
        _last_analysis_time = current_time
        
        # Use optimized real-time analysis
        analysis_data = analyze_realtime(image)
        
        # Create real-time feedback HTML
        feedback_html = create_realtime_feedback_html(analysis_data)
        
        logger.debug(f"Real-time analysis completed - Score: {analysis_data.get('overall_score', 0):.1f}")
        return feedback_html
        
    except Exception as e:
        logger.error(f"Error during real-time analysis: {str(e)}", exc_info=True)
        return f"""
        <div style="text-align: center; padding: 20px; color: #ef4444; background: rgba(239, 68, 68, 0.1); border-radius: 15px;">
            <h3>❌ Analysis Error</h3>
            <p>Unable to analyze image. Please try again.</p>
        </div>
        """


def update_download_visibility(json_file, csv_file, txt_file):
    """Show/hide download files based on analysis completion."""
    has_files = json_file is not None
    return [
        gr.File(visible=has_files),  # JSON
        gr.File(visible=has_files),  # CSV
        gr.File(visible=has_files),  # TXT
    ]


def create_upload_interface() -> gr.Column:
    """Create the upload mode interface."""
    with gr.Column(scale=1) as upload_col:
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
    
    return upload_col, image_input, analyze_btn, download_json, download_csv, download_txt, download_info


def create_live_interface() -> gr.Column:
    """Create the live mode interface with continuous streaming."""
    with gr.Column(scale=1) as live_col:
        # Webcam input with streaming enabled for continuous feed
        webcam_input = gr.Image(
            sources=["webcam"],
            type="pil",
            label="📷 Live Webcam Feed",
            mirror_webcam=False,
            streaming=True  # Enable continuous streaming
        )
        
        # Instructions for live mode
        gr.Markdown("📝 **Live Mode Instructions:**")
        gr.Markdown("1. 📷 Allow webcam access when prompted")
        gr.Markdown("2. 🎯 Position yourself in frame")
        gr.Markdown("3. 📊 Get instant feedback and suggestions every 2 seconds")
        gr.Markdown("4. ✨ Adjust based on real-time tips!")
    
    return live_col, webcam_input


def setup_event_handlers(image_input, analyze_btn, download_json, download_csv, download_txt, 
                        download_info, output_html, webcam_input, webcam_feedback):
    """Set up all event handlers for the interface."""
    
    # Upload mode event handlers
    image_input.change(
        fn=analyze_and_generate_downloads,
        inputs=image_input,
        outputs=[output_html, download_json, download_csv, download_txt, download_info]
    )
    
    analyze_btn.click(
        fn=analyze_and_generate_downloads,
        inputs=image_input,
        outputs=[output_html, download_json, download_csv, download_txt, download_info]
    )
    
    download_json.change(
        fn=update_download_visibility,
        inputs=[download_json, download_csv, download_txt],
        outputs=[download_json, download_csv, download_txt]
    )
    
    # Live mode event handlers - streaming mode for continuous analysis
    logger.info("Setting up webcam streaming with real-time analysis")
    webcam_input.stream(
        fn=analyze_webcam_realtime,
        inputs=webcam_input,
        outputs=webcam_feedback,
        show_progress=False  # Hide progress for smoother experience
    )
    logger.info("Webcam streaming handler configured successfully")
