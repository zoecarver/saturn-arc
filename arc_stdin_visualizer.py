#!/usr/bin/env python3
"""
ARC-AGI Stdin Visualizer
Converts grid data from stdin to PNG images for visual analysis
"""

import sys
import re
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

def parse_grid_from_text(text: str) -> List[List[int]]:
    """
    Parse a grid from text input
    Handles formats like:
    [1, 2, 3]
    [4, 5, 6]
    """
    grid = []
    lines = text.strip().split('\n')
    
    for line in lines:
        # Remove brackets and whitespace
        line = line.strip()
        if not line:
            continue
            
        # Extract numbers from the line
        # This regex finds all sequences of digits
        numbers = re.findall(r'\d+', line)
        
        if numbers:
            row = [int(n) for n in numbers]
            grid.append(row)
    
    return grid

def grid_to_image(grid: List[List[int]], cell_size: int = 30) -> Image.Image:
    """
    Convert a grid to a PNG image
    
    Args:
        grid: 2D list of integers (0-9) representing the grid
        cell_size: Size of each cell in pixels
    
    Returns:
        PIL Image object
    """
    if not grid:
        raise ValueError("Grid is empty")
    
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
    # Parse command line arguments
    output_path = "output.png"
    cell_size = 30
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    if len(sys.argv) > 2:
        cell_size = int(sys.argv[2])
    
    # Read from stdin
    print("Enter grid data (press Ctrl+D when done):", file=sys.stderr)
    input_text = sys.stdin.read()
    
    if not input_text.strip():
        print("Error: No input provided", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse the grid
        grid = parse_grid_from_text(input_text)
        
        if not grid:
            print("Error: Could not parse grid from input", file=sys.stderr)
            sys.exit(1)
        
        # Validate grid dimensions
        width = len(grid[0])
        for i, row in enumerate(grid):
            if len(row) != width:
                print(f"Warning: Row {i+1} has {len(row)} elements, expected {width}", file=sys.stderr)
        
        # Convert to image
        img = grid_to_image(grid, cell_size)
        
        # Save
        img.save(output_path)
        print(f"Saved to {output_path}", file=sys.stderr)
        print(f"Grid size: {len(grid)}x{width}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()