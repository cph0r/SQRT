"""
Report generation module for creating HTML, CSV, and text reports.
"""

import json
import csv
import io
import datetime
from typing import Dict, Any, List, Tuple


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


def save_analysis_results(image, analysis_data: Dict[str, Any]) -> Tuple[str, str, str]:
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
