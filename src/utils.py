"""
Utility functions and Gradio compatibility patches.
"""

import gradio as gr


def apply_gradio_patches():
    """Apply compatibility patches for Gradio with Python 3.13."""
    
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


def get_app_css() -> str:
    """Return CSS styles for the Gradio app."""
    return """
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


def get_app_theme():
    """Return the Gradio theme configuration."""
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    )
