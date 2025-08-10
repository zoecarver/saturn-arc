#!/usr/bin/env python3
"""
ARC-AGI Visualizer
Converts ARC puzzle JSON data to PNG images for visual analysis
"""

import json
import sys
from typing import List
from PIL import Image
import numpy as np

# ARC color palette (0-9 mapped to colors)
ARC_COLORS = {
    0: (0, 0, 0),        # Black (background)
    1: (0, 116, 217),    # Blue
    2: (255, 65, 54),    # Red
    3: (46, 204, 64),    # Green
    4: (255, 220, 0),    # Yellow
    5: (255, 133, 27),   # Orange
    6: (240, 18, 190),   # Magenta/Pink
    7: (127, 219, 255),  # Light Blue/Cyan
    8: (135, 12, 37),    # Dark Red/Maroon
    9: (149, 117, 205),  # Purple
}

def grid_to_image(grid: List[List[int]], cell_size: int = 30) -> Image.Image:
    """
    Convert a grid to a PNG image
    
    Args:
        grid: 2D list of integers (0-9) representing the grid
        cell_size: Size of each cell in pixels
    
    Returns:
        PIL Image object
    """
    height = len(grid)
    width = len(grid[0])
    
    # Create numpy array for the image
    img_array = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
    
    # Fill in the colors
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            color = ARC_COLORS.get(value, (128, 128, 128))
            img_array[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = color
    
    return Image.fromarray(img_array)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc_visualizer.py <input.json> [example_type.index.io] [output.png] [cell_size]")
        print("  example_type: 'train' or 'test'")
        print("  index: 0-based index of the example")
        print("  io: 'input' or 'output'")
        print("  Example: python arc_visualizer.py task.json train.0.input output.png 30")
        sys.exit(1)
    
    # Load JSON
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    # Parse selector (e.g., "train.0.input")
    if len(sys.argv) > 2 and '.' in sys.argv[2]:
        selector = sys.argv[2]
        parts = selector.split('.')
        if len(parts) == 3:
            example_type, index, io_type = parts
            index = int(index)
            
            # Get the specific grid
            grid = data[example_type][index][io_type]
            
            # Get output path
            output_path = sys.argv[3] if len(sys.argv) > 3 else f"{selector}.png"
            
            # Get cell size
            cell_size = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        else:
            print("Invalid selector format. Use: type.index.io (e.g., train.0.input)")
            sys.exit(1)
    else:
        # Default to first training input
        grid = data['train'][0]['input']
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"
        cell_size = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    # Convert to image
    img = grid_to_image(grid, cell_size)
    
    # Save
    img.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()