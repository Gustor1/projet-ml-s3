#!/usr/bin/env python3
"""
scripts/generate_r2_slides.py
Generates slide_r2_title.png and slide_r2_limits.png using PIL to match the style of roles 3 and 4.
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
    draw.text((80, 80), "Role 2 — Audio Preprocessing", fill=(255, 255, 255), font=font_title)
    draw.text((80, 150), "& DSP Architecture", fill=(177, 151, 252), font=font_title)
    
    # Subtitle text
    draw.text((80, 240), "Topic 3: Local Audio Preprocessing for Better ASR Performance", fill=(200, 200, 220), font=font_subtitle)

    # Draw separator line
    draw.line((80, 300, 944, 300), fill=(80, 70, 120), width=2)

    # Draw pipeline diagram header
    draw.text((80, 330), "Preprocessing & Routing Pipeline Flow", fill=(255, 255, 255), font=font_section)

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

    def draw_arrow_right(x1, y1, x2, y2, color=(140, 130, 180)):
        draw.line((x1, y1, x2, y2), fill=color, width=3)
        draw.polygon([(x2, y2), (x2-10, y2-6), (x2-10, y2+6)], fill=color)

    # Box dimensions
    bw, bh = 220, 80

    # Draw Pipeline Steps
    # Raw Input
    draw_box(80, 440, 160, bh, (40, 35, 60), (100, 90, 150), "Raw Audio Input", "Wave/FLAC")
    draw_arrow_right(240, 480, 290, 480)

    # Silence Trim & Norm
    draw_box(290, 440, 240, bh, (30, 50, 80), (70, 120, 180), "DSP Conditioning", "VAD & Peak Norm")
    draw_arrow_right(530, 480, 580, 480)

    # Parallel Stream Routing Box
    draw_box(580, 410, 140, 140, (18, 94, 74), (32, 160, 120), "Parallel Stream", "Routing")
    
    # Split flow
    # ASR Stream
    draw_arrow_right(720, 450, 770, 450)
    draw_box(770, 410, 180, bh, (110, 46, 67), (200, 80, 110), "ASR Pipeline", "Optional Wiener")

    # SER Stream
    draw_arrow_right(720, 510, 770, 510)
    draw_box(770, 490, 180, bh, (20, 70, 50), (40, 150, 100), "SER Pipeline", "Raw Normalized")

    # Additional text at the bottom
    draw.text((80, 680), "Key Achievements:", fill=(177, 151, 252), font=font_section)
    achievements = [
        "• Classical DSP denoising baseline implementation (Wiener, Spectral Subtraction)",
        "• Verification of Whisper hallucination triggers under Spectral Subtraction 'musical noise'",
        "• Verification of SER accuracy degradation from prosody smoothing during denoising",
        "• Decoupled routing pipeline to achieve optimal ASR and SER results concurrently"
    ]
    y_pos = 730
    for ach in achievements:
        draw.text((80, y_pos), ach, fill=(220, 220, 240), font=font_ach)
        y_pos += 40

    os.makedirs("visuals", exist_ok=True)
    img.save("visuals/slide_r2_title.png")
    print("Created visuals/slide_r2_title.png successfully.")

def create_limits_slide():
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
    draw.text((80, 70), "The Limits of Classical DSP Denoising", fill=(255, 255, 255), font=font_title)
    draw.text((80, 130), "Empirical Trade-offs and Degradation Baselines", fill=(244, 67, 54), font=font_title)

    # Separator
    draw.line((80, 200, 944, 200), fill=(100, 40, 40), width=2)

    # Section 1: Spectral Subtraction
    draw.rectangle((80, 240, 944, 530), fill=(30, 20, 25), outline=(150, 50, 60), width=2)
    draw.text((110, 260), "1. Spectral Subtraction & Whisper Hallucinations", fill=(255, 100, 100), font=font_section)
    
    ss_text = (
        "• Methodology: Overlap-add FFT noise magnitude subtraction.\n"
        "• Musical Noise: Introduces random spectral spikes in amplitude.\n"
        "• Impact on Whisper: Attention layers interpret musical noise as phonemes,\n"
        "  leading to transcription hallucinations (WER increased by up to +27.0%).\n"
        "• Result: Kept in API strictly as a documented negative baseline (Evans et al., 2005)."
    )
    draw.text((110, 310), ss_text, fill=(230, 210, 210), font=font_body, spacing=8)

    # Section 2: Wiener Filter
    draw.rectangle((80, 570, 944, 860), fill=(20, 30, 25), outline=(50, 150, 100), width=2)
    draw.text((110, 590), "2. The Wiener Filter & Prosody Destruction", fill=(100, 220, 150), font=font_section)
    
    wiener_text = (
        "• Methodology: Optimal MMSE linear estimator for stationary noise.\n"
        "• Impact on ASR: Improved WER under stationary White Noise (by 2.75% absolute).\n"
        "• Impact on SER: Acts as an 'emotional eraser' by smoothing micro-features\n"
        "  such as pitch variation, jitter, and shimmer.\n"
        "• Result: Dropped Speech Emotion Recognition accuracy by 21.4% (45.8% to 24.4%)."
    )
    draw.text((110, 640), wiener_text, fill=(210, 230, 210), font=font_body, spacing=8)

    # Core Scientific Conclusion at bottom
    draw.rectangle((80, 900, 944, 980), fill=(25, 25, 45), outline=(100, 100, 200), width=2)
    conclusion_label = "Parallel Stream Routing is mandatory. A single preprocessing chain fails both models."
    tw = draw.textlength(conclusion_label, font=font_bold_body)
    draw.text((80 + (864 - tw)/2, 925), conclusion_label, fill=(180, 180, 255), font=font_bold_body)

    img.save("visuals/slide_r2_limits.png")
    print("Created visuals/slide_r2_limits.png successfully.")

if __name__ == "__main__":
    create_title_slide()
    create_limits_slide()
