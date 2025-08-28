#!/usr/bin/env python3
"""
CNN Architecture Visualizer for Binary Photo Classifier

This script creates and visualizes the CNN architecture used in the 
Binary Photo Classifier and File Compressor project using visualkeras.

Usage: python examples/visualize_cnn_architecture.py

Requirements:
- visualkeras
- tensorflow
- PIL (Pillow)

Install: pip install visualkeras tensorflow pillow
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import visualkeras
from PIL import ImageFont

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def create_model():
    """
    Create the exact CNN model architecture used in the project
    Returns the compiled model
    """
    print("Creating CNN model architecture...")
    
    model = keras.Sequential(name="Binary_Photo_Classifier")

    # Layer 1: Feature extraction
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=(150,150,3), name='Conv2D_1'))
    model.add(keras.layers.MaxPool2D(2, 2, name='MaxPool2D_1'))

    # Layer 2: Enhanced feature detection  
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', name='Conv2D_2'))
    model.add(keras.layers.MaxPool2D(2, 2, name='MaxPool2D_2'))

    # Layer 3: Complex pattern recognition
    model.add(keras.layers.Conv2D(128, (3,3), activation='relu', name='Conv2D_3'))
    model.add(keras.layers.MaxPool2D(2, 2, name='MaxPool2D_3'))

    # Layer 4: Deep feature extraction
    model.add(keras.layers.Conv2D(128, (3,3), activation='relu', name='Conv2D_4'))
    model.add(keras.layers.MaxPool2D(2, 2, name='MaxPool2D_4'))

    # Classification layers
    model.add(keras.layers.Flatten(name='Flatten'))
    model.add(keras.layers.Dense(512, activation='relu', name='Dense_Hidden'))
    model.add(keras.layers.Dense(1, activation='sigmoid', name='Dense_Output'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    print("Model created successfully!")
    return model

def create_annotated_architecture_diagram(model, output_dir="assets"):
    """
    Create a custom annotated architecture diagram with detailed information
    """
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors for different layer types
    colors = {
        'input': '#E8F4FD',
        'conv': '#FF6B6B',
        'pool': '#4ECDC4', 
        'dense': '#45B7D1',
        'flatten': '#96CEB4'
    }
    
    # Layer positions and details
    layers = [
        {'name': 'Input Image', 'type': 'input', 'pos': (1, 10), 'size': (1.5, 1), 'details': '150×150×3\n22,500 pixels'},
        {'name': 'Conv2D_1', 'type': 'conv', 'pos': (1, 8.5), 'size': (1.5, 0.8), 'details': '32 filters, 3×3\n896 params\n148×148×32'},
        {'name': 'MaxPool2D_1', 'type': 'pool', 'pos': (3, 8.5), 'size': (1.2, 0.6), 'details': '2×2 pooling\n74×74×32'},
        {'name': 'Conv2D_2', 'type': 'conv', 'pos': (1, 7), 'size': (1.5, 0.8), 'details': '64 filters, 3×3\n18,496 params\n72×72×64'},
        {'name': 'MaxPool2D_2', 'type': 'pool', 'pos': (3, 7), 'size': (1.2, 0.6), 'details': '2×2 pooling\n36×36×64'},
        {'name': 'Conv2D_3', 'type': 'conv', 'pos': (1, 5.5), 'size': (1.5, 0.8), 'details': '128 filters, 3×3\n73,856 params\n34×34×128'},
        {'name': 'MaxPool2D_3', 'type': 'pool', 'pos': (3, 5.5), 'size': (1.2, 0.6), 'details': '2×2 pooling\n17×17×128'},
        {'name': 'Conv2D_4', 'type': 'conv', 'pos': (1, 4), 'size': (1.5, 0.8), 'details': '128 filters, 3×3\n147,584 params\n15×15×128'},
        {'name': 'MaxPool2D_4', 'type': 'pool', 'pos': (3, 4), 'size': (1.2, 0.6), 'details': '2×2 pooling\n7×7×128'},
        {'name': 'Flatten', 'type': 'flatten', 'pos': (5, 4), 'size': (1, 0.6), 'details': '6,272 features\n1D vector'},
        {'name': 'Dense_Hidden', 'type': 'dense', 'pos': (7, 4), 'size': (1.5, 0.8), 'details': '512 neurons\nReLU activation\n3,211,776 params'},
        {'name': 'Dense_Output', 'type': 'dense', 'pos': (7, 2.5), 'size': (1.5, 0.6), 'details': '1 neuron\nSigmoid activation\n513 params'},
    ]
    
    # Draw layers
    for layer in layers:
        x, y = layer['pos']
        w, h = layer['size']
        
        # Create rectangle
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, 
                               linewidth=2, edgecolor='black', 
                               facecolor=colors[layer['type']], alpha=0.8)
        ax.add_patch(rect)
        
        # Add layer name
        ax.text(x, y+h/2-0.1, layer['name'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Add details
        ax.text(x, y-0.1, layer['details'], ha='center', va='center', 
                fontsize=8, style='italic')
    
    # Draw arrows between layers
    arrow_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)
    ]
    
    for i, j in arrow_pairs:
        start = layers[i]['pos']
        end = layers[j]['pos']
        
        if i == 8 and j == 9:  # Special arrow from MaxPool to Flatten
            ax.annotate('', xy=(end[0]-0.5, end[1]), xytext=(start[0]+0.6, start[1]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        elif i == 9 and j == 10:  # Flatten to Dense
            ax.annotate('', xy=(end[0]-0.75, end[1]), xytext=(start[0]+0.5, start[1]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        elif i == 10 and j == 11:  # Dense to Output
            ax.annotate('', xy=(end[0], end[1]+0.3), xytext=(start[0], start[1]-0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        else:
            if start[0] == end[0]:  # Vertical arrow
                ax.annotate('', xy=(end[0], end[1]+layers[j]['size'][1]/2), 
                           xytext=(start[0], start[1]-layers[i]['size'][1]/2),
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
            else:  # Horizontal arrow
                ax.annotate('', xy=(end[0]-layers[j]['size'][0]/2, end[1]), 
                           xytext=(start[0]+layers[i]['size'][0]/2, start[1]),
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # Add title and summary
    plt.title('Binary Photo Classifier - CNN Architecture\n' + 
              f'Total Parameters: {model.count_params():,} | Model Size: ~{model.count_params() * 4 / (1024*1024):.1f} MB | ' +
              'Accuracy: 92.31%', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Layer'),
        patches.Patch(color=colors['conv'], label='Convolutional Layer'),
        patches.Patch(color=colors['pool'], label='MaxPooling Layer'),
        patches.Patch(color=colors['flatten'], label='Flatten Layer'),
        patches.Patch(color=colors['dense'], label='Dense Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add technical specifications box
    specs_text = (
        "Architecture Specifications:\n"
        "• Input: 150×150 RGB images\n"
        "• 4 Conv2D layers with ReLU\n"
        "• Progressive filters: 32→64→128→128\n"
        "• 4 MaxPool2D layers (2×2)\n"
        "• 512-neuron dense layer\n"
        "• Sigmoid output for binary classification\n"
        "• Training Accuracy: 100%\n"
        "• Validation Accuracy: 92.31%"
    )
    
    ax.text(8.5, 8.5, specs_text, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cnn_architecture_annotated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def annotate_visualkeras_image(input_path, output_path, model, diagram_type="basic"):
    """
    Add technical annotations on top of visualkeras generated images
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Open the visualkeras image
    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        if os.name == 'nt':  # Windows
            font_large = ImageFont.truetype('arial.ttf', 24)
            font_medium = ImageFont.truetype('arial.ttf', 18)
            font_small = ImageFont.truetype('arial.ttf', 14)
        else:  # Linux/Mac
            font_large = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
            font_medium = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
            font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Get image dimensions
    width, height = img.size
    
    # Add title with model specifications
    title_text = f"Binary Photo Classifier - CNN Architecture"
    subtitle_text = f"Parameters: {model.count_params():,} | Size: {model.count_params() * 4 / (1024*1024):.1f} MB | Accuracy: 92.31%"
    
    # Add title background
    title_bbox = draw.textbbox((0, 0), title_text, font=font_large)
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=font_medium)
    
    title_bg_height = title_bbox[3] + subtitle_bbox[3] + 30
    draw.rectangle([(0, 0), (width, title_bg_height)], fill='white', outline='black', width=2)
    
    # Draw title and subtitle
    draw.text((width//2, 10), title_text, fill='black', font=font_large, anchor='mt')
    draw.text((width//2, title_bbox[3] + 15), subtitle_text, fill='navy', font=font_medium, anchor='mt')
    
    # Add layer specifications on the right side
    specs_text = [
        "LAYER SPECIFICATIONS:",
        "",
        "Input: 150×150×3 (22,500 pixels)",
        "Conv2D_1: 32×3×3 filters (896 params)",
        "MaxPool_1: 2×2 → 74×74×32",
        "Conv2D_2: 64×3×3 filters (18,496 params)", 
        "MaxPool_2: 2×2 → 36×36×64",
        "Conv2D_3: 128×3×3 filters (73,856 params)",
        "MaxPool_3: 2×2 → 17×17×128", 
        "Conv2D_4: 128×3×3 filters (147,584 params)",
        "MaxPool_4: 2×2 → 7×7×128",
        "Flatten: 6,272 features → 1D",
        "Dense_1: 512 neurons (3,211,776 params)",
        "Dense_2: 1 neuron (513 params)",
        "",
        "ARCHITECTURE FLOW:",
        "150×150×3 → 148×148×32 → 74×74×32",
        "→ 72×72×64 → 36×36×64 → 34×34×128", 
        "→ 17×17×128 → 15×15×128 → 7×7×128",
        "→ 6,272 → 512 → 1"
    ]
    
    # Add specifications box on the right
    spec_x = width - 420
    spec_y = title_bg_height + 20
    spec_box_width = 400
    spec_box_height = len(specs_text) * 20 + 20
    
    # Draw specifications background
    draw.rectangle([(spec_x, spec_y), (spec_x + spec_box_width, spec_y + spec_box_height)], 
                   fill='lightgray', outline='black', width=1)
    
    # Draw specifications text
    for i, line in enumerate(specs_text):
        if line == "LAYER SPECIFICATIONS:" or line == "ARCHITECTURE FLOW:":
            draw.text((spec_x + 10, spec_y + 10 + i * 20), line, fill='darkred', font=font_small)
        elif line == "":
            continue
        else:
            draw.text((spec_x + 10, spec_y + 10 + i * 20), line, fill='black', font=font_small)
    
    # Add parameter breakdown on the left
    param_text = [
        "PARAMETER BREAKDOWN:",
        "",
        f"Conv Layers: {896 + 18496 + 73856 + 147584:,} (7%)",
        f"Dense Layers: {3211776 + 513:,} (93%)", 
        f"Total: {model.count_params():,} parameters",
        "",
        "DESIGN DECISIONS:",
        "• Progressive filter increase",
        "• 3×3 kernel size optimal",
        "• MaxPool for downsampling",
        "• ReLU for non-linearity",
        "• Sigmoid for binary output",
        "• 150×150 input resolution",
        "• Batch size: 5 images"
    ]
    
    # Add parameter box on the left
    param_x = 20
    param_y = title_bg_height + 20
    param_box_width = 300
    param_box_height = len(param_text) * 20 + 20
    
    # Draw parameter background
    draw.rectangle([(param_x, param_y), (param_x + param_box_width, param_y + param_box_height)], 
                   fill='lightyellow', outline='black', width=1)
    
    # Draw parameter text
    for i, line in enumerate(param_text):
        if line == "PARAMETER BREAKDOWN:" or line == "DESIGN DECISIONS:":
            draw.text((param_x + 10, param_y + 10 + i * 20), line, fill='darkblue', font=font_small)
        elif line == "":
            continue
        else:
            draw.text((param_x + 10, param_y + 10 + i * 20), line, fill='black', font=font_small)
    
    # Save the annotated image
    img.save(output_path, 'PNG', dpi=(300, 300))
    return True

def visualize_architecture(model, output_dir="assets"):
    """
    Create visualizations of the CNN architecture
    """
    print("Generating architecture visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    # Create custom annotated diagram first
    try:
        print("Creating detailed annotated architecture diagram...")
        if create_annotated_architecture_diagram(model, output_dir):
            success_count += 1
            print("✅ Annotated diagram with details created successfully")
    except Exception as e:
        print(f"❌ Annotated diagram failed: {e}")
    
    # Basic visualization with annotations
    try:
        print("Creating basic architecture diagram...")
        visualkeras.layered_view(model, 
                               to_file=f'{output_dir}/cnn_architecture_basic_temp.png',
                               legend=True,
                               scale_xy=2,
                               max_z=1000)
        
        # Add annotations to the basic diagram
        annotate_visualkeras_image(f'{output_dir}/cnn_architecture_basic_temp.png', 
                                 f'{output_dir}/cnn_architecture_basic.png', 
                                 model, "basic")
        
        # Remove temp file
        import os
        if os.path.exists(f'{output_dir}/cnn_architecture_basic_temp.png'):
            os.remove(f'{output_dir}/cnn_architecture_basic_temp.png')
        
        success_count += 1
        print("✅ Basic diagram with annotations created successfully")
    except Exception as e:
        print(f"❌ Basic diagram failed: {e}")
    
    # Detailed visualization with custom colors
    try:
        print("Creating detailed architecture diagram...")
        color_map = {
            keras.layers.Conv2D: (255, 107, 107),      # Red for Conv layers
            keras.layers.MaxPool2D: (78, 205, 196),    # Teal for MaxPool layers  
            keras.layers.Dense: (69, 183, 209),        # Blue for Dense layers
            keras.layers.Flatten: (150, 206, 180),     # Green for Flatten layer
        }
        
        visualkeras.layered_view(model,
                               to_file=f'{output_dir}/cnn_architecture_detailed.png',
                               legend=True,
                               color_map=color_map,
                               scale_xy=3,
                               max_z=1000,
                               spacing=50)
        success_count += 1
        print("✅ Detailed diagram created successfully")
    except Exception as e:
        print(f"❌ Detailed diagram failed: {e}")
        try:
            # Try without color map but with annotations
            print("Trying without color map...")
            visualkeras.layered_view(model,
                                   to_file=f'{output_dir}/cnn_architecture_detailed_temp.png',
                                   legend=True,
                                   scale_xy=3,
                                   max_z=1000,
                                   spacing=50)
            
            # Add annotations to the detailed diagram
            annotate_visualkeras_image(f'{output_dir}/cnn_architecture_detailed_temp.png', 
                                     f'{output_dir}/cnn_architecture_detailed.png', 
                                     model, "detailed")
            
            # Remove temp file
            if os.path.exists(f'{output_dir}/cnn_architecture_detailed_temp.png'):
                os.remove(f'{output_dir}/cnn_architecture_detailed_temp.png')
            
            success_count += 1
            print("✅ Detailed diagram with annotations created (without custom colors)")
        except Exception as e2:
            print(f"❌ Detailed diagram completely failed: {e2}")
    
    # Create a graph visualization
    try:
        print("Creating graph visualization...")
        # Try to use a system font for better text rendering
        font_path = None
        if os.name == 'nt':  # Windows
            font_path = 'C:/Windows/Fonts/arial.ttf'
        else:  # Linux/Mac
            font_paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                         '/System/Library/Fonts/Arial.ttf']
            for path in font_paths:
                if os.path.exists(path):
                    font_path = path
                    break
        
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 16)
            visualkeras.graph_view(model,
                                 to_file=f'{output_dir}/cnn_architecture_graph.png',
                                 font=font)
        else:
            visualkeras.graph_view(model,
                                 to_file=f'{output_dir}/cnn_architecture_graph.png')
        success_count += 1
        print("✅ Graph diagram created successfully")
        
    except Exception as e:
        print(f"❌ Graph visualization failed: {e}")
        try:
            # Try without font specification
            print("Trying basic graph view...")
            visualkeras.graph_view(model,
                                 to_file=f'{output_dir}/cnn_architecture_graph.png')
            success_count += 1
            print("✅ Graph diagram created (fallback method)")
        except Exception as e2:
            print(f"❌ Graph visualization completely failed: {e2}")
    
    print(f"\n📊 Successfully created {success_count}/4 visualizations")
    if success_count > 0:
        print(f"Files saved to '{output_dir}/' directory:")
        if os.path.exists(f'{output_dir}/cnn_architecture_annotated.png'):
            print(f"  🎯 cnn_architecture_annotated.png (DETAILED with all specs on image)")
        if os.path.exists(f'{output_dir}/cnn_architecture_basic.png'):
            print(f"  ✅ cnn_architecture_basic.png (basic layered view)")
        if os.path.exists(f'{output_dir}/cnn_architecture_detailed.png'):
            print(f"  ✅ cnn_architecture_detailed.png (detailed with colors)")
        if os.path.exists(f'{output_dir}/cnn_architecture_graph.png'):
            print(f"  ✅ cnn_architecture_graph.png (graph view)")
    return success_count

def print_model_summary(model):
    """
    Print detailed model information
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    print("\nLayer Details:")
    print("-" * 50)
    
    # Build the model to get output shapes
    model.build(input_shape=(None, 150, 150, 3))
    
    for i, layer in enumerate(model.layers):
        try:
            # Get output shape from the layer's output spec
            output_shape = layer.output.shape if hasattr(layer, 'output') else "Unknown"
            
            if hasattr(layer, 'filters'):
                print(f"{i+1:2d}. {layer.name:<15} | Filters: {layer.filters:3d} | "
                      f"Kernel: {layer.kernel_size} | Output: {output_shape}")
            elif hasattr(layer, 'units'):
                print(f"{i+1:2d}. {layer.name:<15} | Units: {layer.units:4d} | "
                      f"Output: {output_shape}")
            else:
                print(f"{i+1:2d}. {layer.name:<15} | Output: {output_shape}")
        except Exception:
            # Fallback for any layer that might cause issues
            layer_type = type(layer).__name__
            print(f"{i+1:2d}. {layer.name:<15} | Type: {layer_type}")
    
    print("\nDetailed Architecture Breakdown:")
    print("-" * 70)
    
    # Create detailed dimension flow analysis
    import numpy as np
    
    print("📐 DIMENSION FLOW THROUGH THE NETWORK:")
    print("=" * 70)
    
    current_shape = (150, 150, 3)
    print(f"Input Image:           {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} pixels")
    
    # Layer 1
    current_shape = (148, 148, 32)  # After 3x3 conv on 150x150
    print(f"After Conv2D_1:        {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    current_shape = (74, 74, 32)    # After 2x2 maxpool
    print(f"After MaxPool2D_1:     {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    
    # Layer 2
    current_shape = (72, 72, 64)    # After 3x3 conv on 74x74
    print(f"After Conv2D_2:        {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    current_shape = (36, 36, 64)    # After 2x2 maxpool
    print(f"After MaxPool2D_2:     {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    
    # Layer 3
    current_shape = (34, 34, 128)   # After 3x3 conv on 36x36
    print(f"After Conv2D_3:        {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    current_shape = (17, 17, 128)   # After 2x2 maxpool
    print(f"After MaxPool2D_3:     {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    
    # Layer 4
    current_shape = (15, 15, 128)   # After 3x3 conv on 17x17
    print(f"After Conv2D_4:        {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    current_shape = (7, 7, 128)     # After 2x2 maxpool
    print(f"After MaxPool2D_4:     {current_shape[0]}×{current_shape[1]}×{current_shape[2]} = {np.prod(current_shape):,} features")
    
    # Flatten and Dense
    flattened_size = 7 * 7 * 128
    print(f"After Flatten:         {flattened_size:,} features (1D vector)")
    print(f"After Dense_Hidden:    512 neurons (fully connected)")
    print(f"Final Output:          1 neuron (binary classification)")
    
    print("\n🔍 LAYER-BY-LAYER ANALYSIS:")
    print("=" * 70)
    
    layer_details = [
        ("Input", "150×150×3", "RGB Image", "22,500 pixels"),
        ("Conv2D_1", "32 filters, 3×3", "Feature Detection", "896 parameters"),
        ("MaxPool2D_1", "2×2 pooling", "Downsampling", "0 parameters"),
        ("Conv2D_2", "64 filters, 3×3", "Enhanced Features", "18,496 parameters"),
        ("MaxPool2D_2", "2×2 pooling", "Downsampling", "0 parameters"),
        ("Conv2D_3", "128 filters, 3×3", "Complex Patterns", "73,856 parameters"),
        ("MaxPool2D_3", "2×2 pooling", "Downsampling", "0 parameters"),
        ("Conv2D_4", "128 filters, 3×3", "Deep Features", "147,584 parameters"),
        ("MaxPool2D_4", "2×2 pooling", "Downsampling", "0 parameters"),
        ("Flatten", "7×7×128 → 6,272", "Vector Conversion", "0 parameters"),
        ("Dense_Hidden", "512 neurons", "Classification Prep", "3,211,776 parameters"),
        ("Dense_Output", "1 neuron", "Binary Output", "513 parameters"),
    ]
    
    for layer_name, config, purpose, params in layer_details:
        print(f"{layer_name:<15} | {config:<20} | {purpose:<18} | {params}")
    
    print(f"\n📊 TOTAL PARAMETERS: {model.count_params():,}")
    print(f"📦 MODEL SIZE: ~{model.count_params() * 4 / (1024*1024):.1f} MB (float32)")
    print(f"⚡ COMPUTATIONAL COMPLEXITY: {model.count_params() / 1_000_000:.2f}M operations")
    
    print("\n🎯 MODEL SPECIFICATIONS FOR MEDIUM BLOG:")
    print("=" * 70)
    print("• Input Resolution: 150×150 RGB images")
    print("• Architecture: 4-layer CNN + 2 Dense layers") 
    print("• Total Depth: 10 layers (8 feature extraction + 2 classification)")
    print("• Filter Progression: 32 → 64 → 128 → 128 (increasing complexity)")
    print("• Receptive Field: Progressive expansion through convolution")
    print("• Parameter Distribution: 93% in final dense layer")
    print("• Activation Functions: ReLU (hidden) + Sigmoid (output)")
    print("• Output: Single probability (0=Photo, 1=Signature)")
    print("• Training Accuracy: 100% | Validation Accuracy: 92.31%")

def generate_medium_blog_content(model):
    """
    Generate clean, copy-paste ready content for Medium blog
    """
    print("\n" + "="*80)
    print("📝 MEDIUM BLOG READY CONTENT - COPY & PASTE")
    print("="*80)
    
    print("\n## CNN Architecture Overview\n")
    print("Our binary classification model uses a 4-layer Convolutional Neural Network")
    print("optimized for distinguishing between photos and signatures. Here's the complete")
    print("architecture breakdown:\n")
    
    print("### Model Specifications")
    print("```")
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Model Size: ~{model.count_params() * 4 / (1024*1024):.1f} MB")
    print(f"Input Shape: 150×150×3 RGB images")
    print(f"Output: Single probability (0=Photo, 1=Signature)")
    print(f"Training Accuracy: 100%")
    print(f"Validation Accuracy: 92.31%")
    print("```\n")
    
    print("### Architecture Flow")
    print("```")
    print("Input (150×150×3) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool")
    print("→ Conv2D(128) → MaxPool → Conv2D(128) → MaxPool → Flatten")
    print("→ Dense(512) → Dense(1) → Sigmoid Output")
    print("```\n")
    
    print("### Dimension Transformation")
    print("| Layer | Input Size | Output Size | Parameters |")
    print("|-------|------------|-------------|------------|")
    print("| Input | 150×150×3 | 150×150×3 | 0 |")
    print("| Conv2D_1 | 150×150×3 | 148×148×32 | 896 |")
    print("| MaxPool_1 | 148×148×32 | 74×74×32 | 0 |")
    print("| Conv2D_2 | 74×74×32 | 72×72×64 | 18,496 |")
    print("| MaxPool_2 | 72×72×64 | 36×36×64 | 0 |")
    print("| Conv2D_3 | 36×36×64 | 34×34×128 | 73,856 |")
    print("| MaxPool_3 | 34×34×128 | 17×17×128 | 0 |")
    print("| Conv2D_4 | 17×17×128 | 15×15×128 | 147,584 |")
    print("| MaxPool_4 | 15×15×128 | 7×7×128 | 0 |")
    print("| Flatten | 7×7×128 | 6,272 | 0 |")
    print("| Dense_1 | 6,272 | 512 | 3,211,776 |")
    print("| Dense_2 | 512 | 1 | 513 |")
    print("\n### Key Architecture Decisions")
    print("- **Progressive Filter Increase**: 32→64→128→128 for hierarchical feature learning")
    print("- **Small Kernel Size**: 3×3 convolutions for optimal feature extraction")
    print("- **Consistent Pooling**: 2×2 MaxPooling for efficient downsampling")
    print("- **ReLU Activation**: For non-linearity and gradient flow")
    print("- **Sigmoid Output**: Perfect for binary classification probabilities")
    print("- **Parameter Efficiency**: 93% of parameters concentrated in final dense layer")
    
    print("\n" + "="*80)
    print("📄 Content above is formatted for Medium - copy and paste directly!")
    print("="*80)

def main():
    """
    Main function to create model and generate visualizations
    """
    print("Binary Photo Classifier - CNN Architecture Visualizer")
    print("="*55)
    
    try:
        # Create the model
        model = create_model()
        
        # Print model summary
        print_model_summary(model)
        
        # Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS...")
        print("="*60)
        success_count = visualize_architecture(model)
        
        # Generate Medium blog content
        generate_medium_blog_content(model)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        
        if success_count > 0:
            print(f"\n🎉 {success_count} visualization(s) created successfully!")
            print("\nThese files can be used in:")
            print("📖 Documentation and project README")
            print("📊 Presentations and technical reports") 
            print("📝 Medium blog posts and articles")
            print("🎓 Educational materials and tutorials")
        else:
            print("\n⚠️  No visualizations were created.")
            print("Please check your visualkeras installation:")
            print("pip install --upgrade visualkeras")
        
    except ImportError as e:
        print(f"\nError: Missing required package - {e}")
        print("\nTo install required packages:")
        print("pip install visualkeras tensorflow pillow")
        print("\nOr install all at once:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please check your environment and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()