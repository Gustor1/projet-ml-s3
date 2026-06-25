#!/usr/bin/env python3
"""
scripts/generate_r6_slides.py
Generates slide_r6_title.png and slide_r6_dashboard.png using PIL to match the style of roles 2, 3, and 4.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_title_slide():
    # Dimensions and Background
    width, height = 1024, 1024
    bg_color = (11, 10, 18) # Dark dark purple/black
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts
    font_path_bold = "C:/Windows/Fonts/segoeuib.ttf"
    font_path_reg = "C:/Windows/Fonts/segoeui.ttf"
    
    font_title = ImageFont.truetype(font_path_bold, 50)
    font_subtitle = ImageFont.truetype(font_path_reg, 26)
    font_section = ImageFont.truetype(font_path_bold, 28)
    font_label = ImageFont.truetype(font_path_bold, 18)
    font_desc = ImageFont.truetype(font_path_reg, 16)
    font_ach = ImageFont.truetype(font_path_reg, 22)

    # Title text
    draw.text((80, 80), "Role 6 — Interactive Demo,", fill=(255, 255, 255), font=font_title)
    draw.text((80, 150), "Visualization & Video", fill=(244, 63, 94), font=font_title)
    
    # Subtitle text
    draw.text((80, 240), "Topic 3: Local Audio Preprocessing for Better ASR Performance", fill=(200, 200, 220), font=font_subtitle)

    # Draw separator line
    draw.line((80, 300, 944, 300), fill=(120, 70, 90), width=2)

    # Draw dashboard layout diagram header
    draw.text((80, 330), "Streamlit Web Dashboard Architecture", fill=(255, 255, 255), font=font_section)

    def draw_box(x, y, w, h, bg, border, text, desc=""):
        # Draw box shadow
        draw.rectangle((x+3, y+3, x+w+3, y+h+3), fill=(5, 5, 8))
        # Draw main box
        draw.rectangle((x, y, x+w, y+h), fill=bg, outline=border, width=2)
        # Text centering
        tw = draw.textlength(text, font=font_label)
        draw.text((x + (w - tw)/2, y + 15), text, fill=(255, 255, 255), font=font_label)
        if desc:
            tdw = draw.textlength(desc, font=font_desc)
            draw.text((x + (w - tdw)/2, y + 42), desc, fill=(200, 200, 220), font=font_desc)

    def draw_arrow_right(x1, y1, x2, y2, color=(180, 130, 150)):
        draw.line((x1, y1, x2, y2), fill=color, width=3)
        draw.polygon([(x2, y2), (x2-10, y2-6), (x2-10, y2+6)], fill=color)

    # Box dimensions
    bw, bh = 220, 80

    # Draw Sidebar / Layout components
    draw_box(80, 440, 180, bh, (40, 35, 60), (120, 90, 150), "Audio Input Source", "RAVDESS / Live / WAV")
    draw_arrow_right(260, 480, 310, 480)

    draw_box(310, 440, 200, bh, (30, 50, 80), (70, 120, 180), "DSP Preprocessing", "None / Wiener / SS")
    draw_arrow_right(510, 480, 560, 480)

    draw_box(560, 410, 180, 140, (110, 46, 67), (200, 80, 110), "Inference Pipeline", "ASR + SER + NLP")
    
    # Split outcomes
    draw_arrow_right(740, 450, 790, 450)
    draw_box(790, 410, 160, bh, (20, 70, 50), (40, 150, 100), "Visual Feedback", "Wave & Spec plots")

    draw_arrow_right(740, 510, 790, 510)
    draw_box(790, 490, 160, bh, (90, 30, 90), (160, 70, 160), "Calibrated Fusion", "Sarcasm Alerts")

    # Achievements
    draw.text((80, 680), "Key Dashboard Features:", fill=(244, 63, 94), font=font_section)
    achievements = [
        "• Interactive local playground built entirely in Streamlit (740 lines of code)",
        "• Modern Glassmorphic Dark UI utilizing custom CSS overrides & Google Fonts",
        "• Real-time Waveform & Spectrogram comparison with fundamental pitch (F0) overlay",
        "• Live sarcasm banner pulsing alert on cross-modal incongruity events",
        "• Interactive sliders for fine-tuning Wiener filters & Spectral Subtraction parameters"
    ]
    y_pos = 730
    for ach in achievements:
        draw.text((80, y_pos), ach, fill=(220, 220, 240), font=font_ach)
        y_pos += 40

    os.makedirs("visuals", exist_ok=True)
    img.save("visuals/slide_r6_title.png")
    print("Created visuals/slide_r6_title.png successfully.")

def create_dashboard_slide():
    # Dimensions and Background
    width, height = 1024, 1024
    bg_color = (11, 10, 18)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts
    font_path_bold = "C:/Windows/Fonts/segoeuib.ttf"
    font_path_reg = "C:/Windows/Fonts/segoeui.ttf"
    
    font_title = ImageFont.truetype(font_path_bold, 38)
    font_section = ImageFont.truetype(font_path_bold, 28)
    font_body = ImageFont.truetype(font_path_reg, 22)
    font_bold_body = ImageFont.truetype(font_path_bold, 22)

    # Title
    draw.text((80, 70), "Interactive Visuals & Multimodal Integration", fill=(255, 255, 255), font=font_title)
    draw.text((80, 130), "Custom CSS System & Pitch-Overlay Spectrograms", fill=(177, 151, 252), font=font_title)

    # Separator
    draw.line((80, 200, 944, 200), fill=(100, 60, 100), width=2)

    # Section 1: UI / CSS
    draw.rectangle((80, 240, 944, 530), fill=(25, 20, 35), outline=(150, 80, 150), width=2)
    draw.text((110, 260), "1. Premium UI & Glassmorphism Design System", fill=(230, 140, 230), font=font_section)
    
    css_text = (
        "• Typography: Injected Google 'Outfit' font family replacing default sans-serif.\n"
        "• Background: Custom 135-degree deep indigo-to-purple dark gradient.\n"
        "• Glassmorphism: Custom metrics cards with backdrop blur & subtle border highlights.\n"
        "• Micro-Animations: Pulsing keyframe red shadow alert for sarcasm detection.\n"
        "• Live Input: Audio input widget for direct recording and immediate feedback loop."
    )
    draw.text((110, 310), css_text, fill=(230, 210, 230), font=font_body, spacing=8)

    # Section 2: Real-time Plotting
    draw.rectangle((80, 570, 944, 860), fill=(20, 25, 35), outline=(70, 120, 180), width=2)
    draw.text((110, 590), "2. Real-Time Signal Processing Visualizations", fill=(100, 180, 255), font=font_section)
    
    viz_text = (
        "• Side-by-Side Analysis: Simultaneous plotting of raw and preprocessed speech waves.\n"
        "• Spectrogram Rendering: Computation of log-frequency spectrograms on the fly.\n"
        "• Pitch Contour Tracking: Dynamic extraction of human fundamental frequency (F0)\n"
        "  using the YIN algorithm (75Hz - 400Hz bound) overlayed as a hot-pink contour.\n"
        "• Hyperparameter Sliders: Immediate updates to signal figures, SNR levels, and latency\n"
        "  upon dragging the filter configuration controls."
    )
    draw.text((110, 640), viz_text, fill=(210, 220, 240), font=font_body, spacing=8)

    # Core Scientific Conclusion at bottom
    draw.rectangle((80, 900, 944, 980), fill=(25, 25, 45), outline=(100, 100, 200), width=2)
    conclusion_label = "Aesthetic dashboards bridge the gap between abstract ML metrics and human perception."
    tw = draw.textlength(conclusion_label, font=font_bold_body)
    draw.text((80 + (864 - tw)/2, 925), conclusion_label, fill=(180, 180, 255), font=font_bold_body)

    img.save("visuals/slide_r6_dashboard.png")
    print("Created visuals/slide_r6_dashboard.png successfully.")

if __name__ == "__main__":
    create_title_slide()
    create_dashboard_slide()
