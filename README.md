---
title: SQRT - Selfie Quality Rater
emoji: 📸
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 📸 SQRT - Selfie Quality Rater ✨

**Professional AI-powered selfie analysis with real-time feedback!**

[![Production Ready](https://img.shields.io/badge/status-production%20ready-success)](https://github.com)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-ready Gradio application that analyzes selfie quality using advanced computer vision techniques, providing comprehensive feedback across multiple dimensions including face detection, technical quality, composition, and style.

## 🚀 Features

### 📊 **Comprehensive Analysis**
- **Face Detection & Quality**: Automatic face recognition, positioning, size, and lighting analysis
- **Technical Metrics**: Sharpness, lighting, contrast, and background complexity
- **Advanced Analysis**: Emotion detection, eye contact, composition (rule of thirds)
- **Style Assessment**: Color harmony, outfit coordination, beauty filter detection
- **Lighting Analysis**: Time-of-day detection and optimal lighting conditions

### 🎯 **Dual Analysis Modes**
- **📤 Upload Mode**: Detailed analysis with downloadable reports (JSON, CSV, TXT)
- **📹 Live Mode**: Real-time webcam streaming with continuous analysis every 2 seconds

### 🧠 **Intelligent Feedback**
- Context-aware suggestions prioritized by impact
- Personalized improvement recommendations
- Professional photography tips and techniques

## 🏗️ Architecture

### Modular Design
The application is built with a clean, modular architecture for maintainability and extensibility:

```
SQRT/
├── app.py                     # Main application entry point (95 lines)
├── requirements.txt           # Python dependencies
└── src/                      # Modular source code
    ├── face_detection.py     # Face detection and analysis
    ├── image_analysis.py     # Technical image quality metrics
    ├── advanced_analysis.py  # Emotion, composition, style analysis
    ├── analyzer.py          # Main analysis orchestrator
    ├── suggestion_generator.py # Intelligent feedback system
    ├── report_generator.py  # HTML, CSV, and text reports
    ├── ui_components.py     # Gradio interface components
    └── utils.py             # Utilities and compatibility patches
```

### 📦 **Module Overview**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `face_detection.py` | Face detection and face-specific analysis | `detect_faces()`, `analyze_face_quality()`, `analyze_eye_contact()` |
| `image_analysis.py` | Technical image quality metrics | `analyze_sharpness()`, `analyze_lighting()`, `analyze_contrast()` |
| `advanced_analysis.py` | Advanced computer vision analysis | `analyze_emotion()`, `analyze_rule_of_thirds()`, `analyze_beauty_filter()` |
| `analyzer.py` | Main analysis orchestrator | `score_selfie()`, `analyze_realtime()` |
| `suggestion_generator.py` | Intelligent feedback generation | `generate_suggestions()` |
| `report_generator.py` | Report creation and export | `create_score_html()`, `save_analysis_results()` |
| `ui_components.py` | Gradio interface components | `create_upload_interface()`, `create_live_interface()` |
| `utils.py` | Utilities and compatibility | `apply_gradio_patches()`, theme/CSS functions |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Webcam (optional, for live mode)

### Quick Start

1. **Clone and navigate to the repository**
   ```bash
   git clone <repository-url>
   cd SQRT
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   - Navigate to the URL printed in the terminal (typically `http://127.0.0.1:7860`)
   - Choose between Upload Mode or Live Mode
   - Start analyzing your selfies!

## 📋 Dependencies

- **gradio**: Web interface framework
- **opencv-python**: Computer vision and image processing
- **pillow**: Image manipulation
- **numpy**: Numerical computing
- **audioop-lts**: Audio processing compatibility

## 🎮 Usage Guide

### 📤 **Upload Mode**
1. Click the "📤 Upload Mode" tab
2. Upload a selfie image using the file uploader
3. View comprehensive analysis results with scores and suggestions
4. Download detailed reports in JSON, CSV, or TXT format

### 📹 **Live Mode**
1. Click the "📹 Live Mode" tab
2. Allow camera access when prompted
3. Click the camera button to take a selfie
4. Get instant analysis and feedback
5. Retake photos to see improvements in real-time

## 📊 Analysis Categories

### 🔍 **Technical Quality (4 metrics)**
- **Lighting**: Exposure balance and illumination quality
- **Sharpness**: Focus clarity using edge detection algorithms
- **Contrast**: Dynamic range and visual depth
- **Background**: Clutter analysis and subject isolation

### 👤 **Face Analysis (5 metrics)** *(when face detected)*
- **Face Detection**: Automatic face recognition
- **Face Positioning**: Centering and composition
- **Face Size**: Optimal framing distance
- **Face Lighting**: Face-specific illumination
- **Eye Contact**: Gaze direction analysis

### 🧠 **Advanced Analysis (6 metrics)**
- **Emotion Quality**: Facial expression analysis
- **Color Harmony**: Color temperature and balance
- **Composition**: Rule of thirds compliance
- **Naturalness**: Beauty filter and over-processing detection
- **Style**: Outfit coordination and pattern analysis
- **Lighting Time**: Optimal shooting conditions

## 🔧 Development

### Adding New Analysis Features

The modular architecture makes it easy to extend functionality:

```python
# Add new analysis to advanced_analysis.py
def analyze_new_feature(image_array, faces):
    # Your analysis logic here
    return {"new_metric_score": score}

# Update analyzer.py to include new analysis
def score_selfie(image):
    # ... existing code ...
    new_data = analyze_new_feature(img_array, faces)
    scores["new_metric"] = new_data["new_metric_score"]
```

### Testing Individual Modules

```bash
# Test specific modules
python3 -c "from src.face_detection import detect_faces; print('Face detection OK')"
python3 -c "from src.analyzer import score_selfie; print('Analyzer OK')"
```

## 📈 Technical Details

### Face Detection
- Uses OpenCV's Haar Cascade classifiers
- Supports both frontal and profile face detection
- Advanced filtering to reduce false positives
- Face quality analysis including positioning and lighting

### Image Analysis
- Laplacian variance for sharpness measurement
- Histogram analysis for lighting quality
- Edge density for background complexity assessment
- Color space analysis for harmony and temperature

### Real-time Performance
- Optimized processing pipeline for live analysis
- Image resizing for faster computation
- Essential metrics only for real-time feedback

## 🔒 Privacy & Security

- **Local Processing**: All analysis happens on your device
- **No Data Upload**: Images are not sent to external servers
- **No Storage**: Photos are not saved unless you explicitly download reports
- **Open Source**: Full transparency of analysis algorithms

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**Camera Access Issues (Live Mode)**
- Ensure camera permissions are granted in your browser
- Try refreshing the page if camera doesn't appear
- Check if other applications are using the camera

**Performance Issues**
- Close other applications using significant CPU/GPU
- Reduce browser zoom level for better UI performance
- Use Upload Mode for very high-resolution images

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

Deploy SQRT to Hugging Face Spaces with automated CI/CD:

1. **Quick Deploy**:
   - See detailed instructions in [DEPLOYMENT.md](DEPLOYMENT.md)
   - Automatic deployment on every push to `main` branch
   - Free hosting with GPU acceleration available

2. **Requirements**:
   - GitHub repository
   - Hugging Face account
   - Hugging Face access token (Write permissions)

3. **Setup**:
   ```bash
   # 1. Create a Space on Hugging Face
   # 2. Add GitHub secrets: HF_TOKEN and HF_SPACE_NAME
   # 3. Push to main branch - automatic deployment!
   ```

4. **Access your deployed app**:
   - `https://huggingface.co/spaces/YOUR_USERNAME/sqrt-selfie-rater`

For complete deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Local Deployment

For production-ready local deployment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn (production server)
pip install gunicorn
gunicorn app:main -b 0.0.0.0:7860 -w 4

# Or use the built-in Gradio server
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements to the appropriate module
4. Test your changes
5. Submit a pull request

## 📄 License

See `LICENSE` file for details.

## 🙏 Acknowledgments

- OpenCV team for computer vision algorithms
- Gradio team for the excellent web interface framework
- The open-source community for inspiration and tools

---

**Built with ❤️ using advanced computer vision and machine learning techniques**

*SQRT rates photo quality, not the person. Our goal is to help you take better selfies through constructive, technical feedback.*