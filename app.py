#!/usr/bin/env python3
"""
SQRT - Selfie Quality Rater
A modular Gradio application for analyzing selfie quality using computer vision.
"""

import gradio as gr
from src.utils import apply_gradio_patches, get_app_css, get_app_theme
from src.ui_components import (
    create_upload_interface, create_live_interface, setup_event_handlers
)

# Apply compatibility patches first
apply_gradio_patches()


def main() -> None:
    """Main application entry point."""
    with gr.Blocks(
        title="Selfie Quality Rater (SQRT)",
        theme=get_app_theme(),
        css=get_app_css()
    ) as demo:
        
        # App header and description
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
                    # Create upload interface
                    upload_col, image_input, analyze_btn, download_json, download_csv, download_txt, download_info = create_upload_interface()
                    
                    # Results display
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
                    # Create live interface
                    live_col, webcam_input = create_live_interface()
                    
                    # Live feedback display
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
        
        # Set up all event handlers
        setup_event_handlers(
            image_input, analyze_btn, download_json, download_csv, download_txt,
            download_info, output_html, webcam_input, webcam_feedback
        )
        
        # App footer with feature information
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
    
    # Launch the application
    demo.launch(share=True, show_api=False, show_error=True)


if __name__ == "__main__":
    main()
