#!/usr/bin/env python3
"""
Demo Enhanced Report Generator

Creates a sample enhanced HTML report showing what the full pipeline would produce:
- Full SmolDocling content
- Extracted images  
- Three VLM model comparisons
- Contextual chunks
"""

import json
from pathlib import Path
from datetime import datetime
import base64

def create_demo_enhanced_report():
    """Create a demonstration enhanced HTML report"""
    
    # Create demo data structure
    demo_data = {
        "document": "Preview_BMW_X5_G05.pdf",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "pages": 17,
            "segments": 67,
            "visual_elements": 50,
            "chunks": 9,
            "processing_time": 45.2
        },
        "pages": [
            {
                "page_num": 1,
                "segments": [
                    {
                        "type": "heading",
                        "content": "BMW X5 (C-SUV), ME: November 2018, PA: April 2023",
                        "char_count": 48
                    },
                    {
                        "type": "text",
                        "content": """Update: X5 drive 45e (PHEV)
Update: X5 M / X5 M Competition
AUDI AG I/EG-A12 Juni 2018

Die neue Generation des BMW X5 steht für souveräne Fahrfreude auf jedem Terrain. 
Das Sports Activity Vehicle kombiniert robuste Geländetauglichkeit mit dem 
unverwechselbaren Fahrverhalten eines BMW auf der Straße.""",
                        "char_count": 287
                    }
                ],
                "visual_elements": [
                    {
                        "type": "image",
                        "description_qwen": "The image shows a blue BMW X5 SUV driving on a mountain road. The vehicle is captured in motion against a scenic backdrop of mountains and forest. The SUV features BMW's characteristic kidney grille, LED headlights, and sporty alloy wheels. The setting appears to be in a mountainous region with winding roads.",
                        "description_llava": "This image depicts a BMW X5 sports utility vehicle in motion on what appears to be a mountain pass. The vehicle is painted in a metallic blue color and shows the characteristic design elements of BMW's luxury SUV line, including the prominent front grille, angular headlights, and muscular wheel arches. The background suggests an alpine or mountainous environment with trees and peaks visible.",
                        "description_pixtral": "A luxury SUV navigating a curved mountain road surrounded by natural scenery.",
                        "confidence_qwen": 0.92,
                        "confidence_llava": 0.95,
                        "confidence_pixtral": 0.78,
                        "ocr_qwen": "M-BY 5173"
                    }
                ]
            },
            {
                "page_num": 2,
                "segments": [
                    {
                        "type": "table",
                        "content": """Technische Daten - BMW X5 Modelljahr 2023
┌─────────────────┬─────────────┬──────────────┬───────────┐
│ Modell          │ Motor       │ Leistung     │ Preis     │
├─────────────────┼─────────────┼──────────────┼───────────┤
│ X5 xDrive40i    │ R6 3,0l     │ 280 kW       │ 81.900 €  │
│ X5 xDrive45e    │ R6 3,0l PHEV│ 290 kW       │ 92.400 €  │
│ X5 M60i xDrive  │ V8 4,4l     │ 390 kW       │ 116.700 € │
└─────────────────┴─────────────┴──────────────┴───────────┘""",
                        "char_count": 412
                    }
                ],
                "visual_elements": [
                    {
                        "type": "chart",
                        "description_qwen": "This infographic displays the BMW X5 model lineup comparison chart. It shows different trim levels (xDrive40i, xDrive45e, M60i) with their respective engine specifications, power outputs in kW, fuel consumption figures, and base prices in Euros. The chart uses BMW's corporate blue color scheme with clear typography for easy readability.",
                        "description_llava": "The image contains a detailed comparison table for BMW X5 variants. The table is structured with columns for model designation, engine type (including a plug-in hybrid option), power output, fuel consumption, and pricing. Notable is the inclusion of the PHEV variant (xDrive45e) positioned between the base gasoline model and the high-performance M60i variant.",
                        "description_pixtral": "A technical specifications table showing BMW vehicle data.",
                        "confidence_qwen": 0.88,
                        "confidence_llava": 0.91,
                        "confidence_pixtral": 0.65
                    }
                ]
            },
            {
                "page_num": 3,
                "segments": [
                    {
                        "type": "heading",
                        "content": "Interieur und Konnektivität",
                        "char_count": 27
                    },
                    {
                        "type": "text", 
                        "content": """Das Interieur des neuen BMW X5 definiert moderne Luxusklasse neu. 
Das BMW Curved Display vereint ein 12,3" Informationsdisplay mit einem 
14,9" Control Display zu einer eindrucksvollen Einheit. Die neueste 
Generation des BMW iDrive mit Operating System 8 bietet intuitive 
Bedienung per Touch, Sprache oder BMW Controller.""",
                        "char_count": 324
                    }
                ],
                "visual_elements": [
                    {
                        "type": "image",
                        "description_qwen": "The interior image showcases the BMW X5's luxurious cabin featuring the impressive BMW Curved Display that seamlessly integrates the digital instrument cluster and central infotainment screen. The dashboard shows a modern, minimalist design with high-quality materials including leather upholstery, ambient lighting strips, and metallic accents. The center console features the crystal-finished iDrive controller and gear selector.",
                        "description_llava": "This photograph captures the driver-focused interior of the BMW X5. Prominent is the curved display panel that houses both the driver information display and the central touchscreen. The interior demonstrates BMW's commitment to luxury with visible premium materials such as perforated leather seats, wood or carbon fiber trim options, and the signature ambient lighting system that creates a sophisticated atmosphere.",
                        "description_pixtral": "A modern vehicle interior with digital displays and premium materials.",
                        "confidence_qwen": 0.94,
                        "confidence_llava": 0.96,
                        "confidence_pixtral": 0.72
                    }
                ]
            }
        ],
        "chunks": [
            {
                "id": "chunk_001",
                "type": "header",
                "content": "BMW X5 (C-SUV), ME: November 2018, PA: April 2023\nUpdate: X5 drive 45e (PHEV)\nUpdate: X5 M / X5 M Competition",
                "token_count": 45,
                "inherited_context": "Document: BMW X5 Technical Preview"
            },
            {
                "id": "chunk_002",
                "type": "content",
                "content": "Die neue Generation des BMW X5 steht für souveräne Fahrfreude auf jedem Terrain. Das Sports Activity Vehicle kombiniert robuste Geländetauglichkeit mit dem unverwechselbaren Fahrverhalten eines BMW auf der Straße.",
                "token_count": 78,
                "inherited_context": "Section: Overview | Document: BMW X5 Technical Preview"
            }
        ]
    }
    
    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Pipeline Analysis - {demo_data['document']}</title>
    <style>
        {get_enhanced_styles()}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Enhanced Pipeline Analysis Report</h1>
        <p>{demo_data['document']} - {demo_data['timestamp']}</p>
    </div>
    
    <div class="container">
        <!-- Summary Section -->
        <div class="section">
            <h2>📊 Processing Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>⏱️ Performance</h3>
                    <p><strong>Total Time:</strong> {demo_data['stats']['processing_time']}s</p>
                    <p><strong>SmolDocling:</strong> 38.2s</p>
                    <p><strong>VLM Processing:</strong> 5.8s</p>
                    <p><strong>Chunking:</strong> 1.2s</p>
                </div>
                
                <div class="stat-card">
                    <h3>📄 Document Stats</h3>
                    <p><strong>Pages:</strong> {demo_data['stats']['pages']}</p>
                    <p><strong>Segments:</strong> {demo_data['stats']['segments']}</p>
                    <p><strong>Visual Elements:</strong> {demo_data['stats']['visual_elements']}</p>
                    <p><strong>Chunks:</strong> {demo_data['stats']['chunks']}</p>
                </div>
                
                <div class="stat-card">
                    <h3>🤖 VLM Analysis</h3>
                    <p><strong>Processed Elements:</strong> {demo_data['stats']['visual_elements']}</p>
                    <p><strong>Models Used:</strong> Qwen2.5-VL, LLaVA, Pixtral</p>
                    <p><strong>Avg Confidence:</strong> 87.3%</p>
                </div>
            </div>
        </div>
        
        <!-- Page-by-Page Analysis -->
"""
    
    # Add page sections
    for page in demo_data['pages']:
        html_content += f"""
        <div class="section page-section">
            <h2>📄 Page {page['page_num']} Analysis</h2>
            
            <!-- SmolDocling Content -->
            <div class="smoldocling-content">
                <h3>📝 Extracted Content (SmolDocling)</h3>
"""
        
        # Add segments
        for segment in page['segments']:
            segment_class = f"{segment['type']}-segment"
            html_content += f"""
                <div class="segment {segment_class}">
                    <div class="segment-header">
                        <span class="segment-type">{segment['type']}</span>
                        <span class="segment-chars">{segment['char_count']} chars</span>
                    </div>
                    <div class="segment-content">
                        {segment['content']}
                    </div>
                </div>
"""
        
        html_content += """
            </div>
"""
        
        # Add visual elements with VLM comparisons
        if page.get('visual_elements'):
            html_content += """
            <!-- Visual Elements with VLM -->
            <div class="visual-elements-section">
                <h3>🖼️ Visual Elements with VLM Analysis</h3>
"""
            
            for idx, visual in enumerate(page['visual_elements']):
                html_content += f"""
                <div class="visual-element-container">
                    <!-- Visual Image (Demo Placeholder) -->
                    <div class="visual-image">
                        <div class="demo-image-placeholder">
                            <div class="demo-image-text">BMW X5 Image {page['page_num']}-{idx+1}</div>
                            <div class="demo-image-info">{visual['type'].upper()}</div>
                        </div>
                    </div>
                    
                    <!-- VLM Comparisons Grid -->
                    <div class="vlm-comparison-grid">
                        <!-- Qwen2.5-VL -->
                        <div class="vlm-card success">
                            <div class="vlm-header">
                                <h4>🎯 Qwen2.5-VL-7B</h4>
                                <span class="confidence">Confidence: {visual['confidence_qwen']:.0%}</span>
                            </div>
                            <div class="vlm-content">
                                <div class="description">
                                    <strong>Description:</strong><br>
                                    {visual['description_qwen']}
                                </div>
"""
                if visual.get('ocr_qwen'):
                    html_content += f"""
                                <div class="ocr-text">
                                    <strong>OCR Text:</strong><br>
                                    <code>{visual['ocr_qwen']}</code>
                                </div>
"""
                html_content += f"""
                            </div>
                            <div class="vlm-footer">
                                <span class="processing-time">⏱️ 1.8s</span>
                            </div>
                        </div>
                        
                        <!-- LLaVA -->
                        <div class="vlm-card success">
                            <div class="vlm-header">
                                <h4>👁️ LLaVA-1.6-Mistral-7B</h4>
                                <span class="confidence">Confidence: {visual['confidence_llava']:.0%}</span>
                            </div>
                            <div class="vlm-content">
                                <div class="description">
                                    <strong>Description:</strong><br>
                                    {visual['description_llava']}
                                </div>
                            </div>
                            <div class="vlm-footer">
                                <span class="processing-time">⏱️ 2.1s</span>
                            </div>
                        </div>
                        
                        <!-- Pixtral -->
                        <div class="vlm-card success">
                            <div class="vlm-header">
                                <h4>🔍 Pixtral-12B</h4>
                                <span class="confidence">Confidence: {visual['confidence_pixtral']:.0%}</span>
                            </div>
                            <div class="vlm-content">
                                <div class="description">
                                    <strong>Description:</strong><br>
                                    {visual['description_pixtral']}
                                </div>
                            </div>
                            <div class="vlm-footer">
                                <span class="processing-time">⏱️ 1.2s</span>
                            </div>
                        </div>
                    </div>
                </div>
"""
            
            html_content += """
            </div>
"""
        
        html_content += """
        </div>
"""
    
    # Add chunks section
    html_content += """
        <!-- Chunks Analysis -->
        <div class="section">
            <h2>📦 Contextual Chunks Analysis</h2>
            <div class="chunks-grid">
"""
    
    for chunk in demo_data['chunks']:
        html_content += f"""
                <div class="chunk-card">
                    <h4>Chunk {chunk['id']}</h4>
                    <div class="chunk-meta">
                        <span>📦 Type: {chunk['type']}</span>
                        <span>🔢 Tokens: {chunk['token_count']}</span>
                    </div>
                    <div class="chunk-content">
                        {chunk['content']}
                    </div>
                    <div class="context-inheritance">
                        <strong>🔗 Inherited Context:</strong>
                        <div class="inherited-content">
                            {chunk['inherited_context']}
                        </div>
                    </div>
                </div>
"""
    
    html_content += """
            </div>
        </div>
    </div>
    
    <script>
        // Simple image zoom functionality
        document.addEventListener('DOMContentLoaded', function() {
            const images = document.querySelectorAll('.demo-image-placeholder');
            images.forEach(img => {
                img.style.cursor = 'zoom-in';
                img.addEventListener('click', function() {
                    if (this.style.transform === 'scale(1.5)') {
                        this.style.transform = 'scale(1)';
                        this.style.cursor = 'zoom-in';
                    } else {
                        this.style.transform = 'scale(1.5)';
                        this.style.cursor = 'zoom-out';
                    }
                });
            });
        });
    </script>
</body>
</html>"""
    
    return html_content

def get_enhanced_styles():
    """Get enhanced CSS styles for the demo report"""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .section {
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .page-section {
            border-left: 4px solid #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .stat-card h3 {
            margin-top: 0;
            color: #667eea;
        }
        
        /* SmolDocling Content */
        .smoldocling-content {
            margin-bottom: 2rem;
        }
        
        .segment {
            background: #f8f9fa;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border-left: 3px solid #e9ecef;
        }
        
        .heading-segment {
            border-left-color: #667eea;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .table-segment {
            border-left-color: #28a745;
            font-family: monospace;
            font-size: 0.9rem;
        }
        
        .segment-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .segment-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        /* Visual Elements */
        .visual-elements-section {
            margin-top: 2rem;
        }
        
        .visual-element-container {
            background: #f8f9fa;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        
        .visual-image {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        /* Demo Image Placeholder */
        .demo-image-placeholder {
            display: inline-block;
            width: 600px;
            max-width: 100%;
            height: 400px;
            background: linear-gradient(135deg, #e0e0e0 25%, #f0f0f0 25%, #f0f0f0 50%, #e0e0e0 50%, #e0e0e0 75%, #f0f0f0 75%, #f0f0f0);
            background-size: 20px 20px;
            border: 2px solid #ddd;
            border-radius: 8px;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: transform 0.3s ease;
        }
        
        .demo-image-text {
            background: rgba(255, 255, 255, 0.9);
            padding: 1rem 2rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 1.2rem;
            color: #667eea;
        }
        
        .demo-image-info {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #666;
        }
        
        /* VLM Comparison Grid */
        .vlm-comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1rem;
        }
        
        .vlm-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            transition: box-shadow 0.3s ease;
        }
        
        .vlm-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .vlm-card.success {
            border-color: #28a745;
        }
        
        .vlm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e9ecef;
        }
        
        .vlm-header h4 {
            margin: 0;
            color: #667eea;
        }
        
        .confidence {
            font-size: 0.9rem;
            color: #28a745;
            font-weight: bold;
        }
        
        .vlm-content {
            margin-bottom: 1rem;
        }
        
        .description {
            margin-bottom: 1rem;
            line-height: 1.5;
        }
        
        .ocr-text {
            background: #f8f9fa;
            padding: 0.5rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .ocr-text code {
            font-family: monospace;
            font-size: 0.9rem;
            color: #d73a49;
        }
        
        .vlm-footer {
            text-align: right;
            font-size: 0.9rem;
            color: #666;
        }
        
        /* Chunks */
        .chunks-grid {
            display: grid;
            gap: 1rem;
        }
        
        .chunk-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 3px solid #764ba2;
        }
        
        .chunk-meta {
            display: flex;
            gap: 1rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            color: #666;
        }
        
        .chunk-content {
            background: white;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
        }
        
        .context-inheritance {
            background: #e7f3ff;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 4px;
        }
        
        .inherited-content {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            font-style: italic;
        }
    """

def main():
    """Generate demo enhanced report"""
    output_dir = Path("data/debug/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    html_content = create_demo_enhanced_report()
    
    # Save report
    output_file = output_dir / f"BMW_X5_enhanced_demo_{datetime.now():%Y%m%d_%H%M%S}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Demo enhanced report created: {output_file}")
    print(f"📊 Open in browser: file://{output_file.absolute()}")

if __name__ == "__main__":
    main()