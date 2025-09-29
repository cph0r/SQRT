"""
Gradio UI components and interface definitions.
"""

import gradio as gr
import datetime
import json
from PIL import Image
from typing import Tuple

from .analyzer import score_selfie, analyze_realtime
from .report_generator import create_score_html, create_realtime_feedback_html, save_analysis_results


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


def analyze_webcam_selfie(image: Image.Image) -> str:
    """Auto-analyze webcam captures for live mode."""
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
    """Create the live mode interface."""
    with gr.Column(scale=1) as live_col:
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
    
    # Live mode event handlers
    print("🔧 Setting up webcam_input.change event handler...")
    webcam_input.change(
        fn=analyze_webcam_selfie,
        inputs=webcam_input,
        outputs=webcam_feedback,
        show_progress="full"
    )
    print("✅ Webcam event handler set up successfully!")
