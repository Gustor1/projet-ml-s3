import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

def create_slide(filename, title, text_lines, chart_data=None):
    width, height = 1920, 1080
    img = Image.new('RGB', (width, height), color=(30, 30, 40))
    d = ImageDraw.Draw(img)
    
    # Title
    try:
        font_title = ImageFont.truetype("arial.ttf", 80)
        font_text = ImageFont.truetype("arial.ttf", 50)
    except IOError:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    d.text((100, 100), title, font=font_title, fill=(255, 200, 100))
    
    y = 300
    for line in text_lines:
        d.text((100, y), line, font=font_text, fill=(200, 220, 255))
        y += 80

    os.makedirs('visuals', exist_ok=True)
    
    if chart_data:
        # Create a matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e1e28')
        ax.set_facecolor('#1e1e28')
        
        labels, values = zip(*chart_data)
        ax.bar(labels, values, color=['#ff9999','#66b3ff','#99ff99'])
        ax.tick_params(colors='white', labelsize=14)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        chart_path = 'visuals/temp_chart.png'
        plt.savefig(chart_path, bbox_inches='tight', transparent=True)
        plt.close()
        
        chart_img = Image.open(chart_path)
        img.paste(chart_img, (800, 300), chart_img)
        os.remove(chart_path)

    img.save(f'visuals/{filename}')
    print(f"Generated {filename}")

if __name__ == "__main__":
    create_slide(
        'slide_r5_title.png',
        'Role 5: Performance Profiling & Optimization',
        [
            '- Can a 3-model pipeline run in real-time on a CPU?',
            '- Benchmarking unoptimized FP32 PyTorch baseline.',
            '- Goal: Prove viability for Edge Deployment.'
        ]
    )

    create_slide(
        'slide_r5_latency_chart.png',
        'Inference Latency Results (CPU)',
        [
            '- Total Inference Time: 1.3s',
            '- Audio Length: 3.0s',
            '- Real-Time Factor (RTF): 0.4x',
            '- Processing is 2.5x faster than real-time.'
        ],
        chart_data=[('Whisper ASR', 0.84), ('Wav2Vec2 SER', 0.38), ('DistilBERT NLP', 0.09)]
    )

    create_slide(
        'slide_r5_loading_issue.png',
        'The Loading Bottleneck',
        [
            '- Total memory footprint during inference: < 10 MB (Lean)',
            '- BUT Model Initialization takes > 26 seconds.',
            '- Unacceptable cold-start delay for interactive apps.',
            '- Next Step: INT8 Dynamic Quantization (Elio).'
        ]
    )
