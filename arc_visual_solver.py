#!/usr/bin/env python3
"""
ARC-AGI Visual Solver
Uses a phased approach with visual representations to solve ARC puzzles
"""

import json
import os
import sys
import base64
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from PIL import Image
import numpy as np

# Import the visualizer functions
from arc_visualizer import grid_to_image, ARC_COLORS

# Define the visualization tool for function calling
VISUALIZATION_TOOL = {
    "type": "function",
    "name": "visualize_grid",
    "description": "Generate a visual image representation of a grid. Use this to better understand patterns and transformations in the puzzle.",
    "parameters": {
        "type": "object",
        "properties": {
            "grid": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "description": "2D array of integers (0-9) representing the grid to visualize"
            }
        },
        "required": ["grid"]
    }
}

# Define the diff tool for comparing two grids numerically
DIFF_TOOL = {
    "type": "function",
    "name": "diff_grids",
    "description": "Compare two grids numerically and return the differences. Returns which cells are not the same, their positions, and values in each grid.",
    "parameters": {
        "type": "object",
        "properties": {
            "grid1": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "description": "First grid to compare (e.g., input or predicted)"
            },
            "grid2": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "description": "Second grid to compare (e.g., expected output)"
            }
        },
        "required": ["grid1", "grid2"]
    }
}


class ARCVisualSolver:
    def __init__(self):
        """Initialize the ARC visual solver with API credentials"""
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        if not self.api_key_openai:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self.client_openai = OpenAI(api_key=self.api_key_openai)
        
        self.conversation_history = []
        self.current_task_name = None
        
        # Use img_tmp directory in project root
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_tmp")
    
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from a JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display in the prompt"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def create_grid_image(self, grid: List[List[int]], cell_size: int = 30, label: str = "grid") -> str:
        """Create an image from a grid and return the file path"""
        img = grid_to_image(grid, cell_size)
        # Save to temp file with meaningful name including task name
        file_count = len(os.listdir(self.temp_dir))
        if self.current_task_name:
            temp_path = os.path.join(self.temp_dir, f"{self.current_task_name}_{label}_{file_count:03d}.png")
        else:
            temp_path = os.path.join(self.temp_dir, f"{label}_{file_count:03d}.png")
        img.save(temp_path)
        return temp_path
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def diff_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Compare two grids and return differences"""
        differences = []
        rows1, cols1 = len(grid1), len(grid1[0]) if grid1 else 0
        rows2, cols2 = len(grid2), len(grid2[0]) if grid2 else 0
        
        # Find differences in overlapping area
        min_rows = min(rows1, rows2)
        min_cols = min(cols1, cols2)
        
        for row in range(min_rows):
            for col in range(min_cols):
                if grid1[row][col] != grid2[row][col]:
                    differences.append({
                        "position": [row, col],
                        "grid1_value": grid1[row][col],
                        "grid2_value": grid2[row][col]
                    })
        
        # Extra rows in grid1 (if rows1 > rows2)
        for row in range(min_rows, rows1):
            for col in range(cols1):
                differences.append({
                    "position": [row, col],
                    "grid1_value": grid1[row][col],
                    "grid2_value": "N/A (row doesn't exist)"
                })
        
        # Extra rows in grid2 (if rows2 > rows1)
        for row in range(min_rows, rows2):
            for col in range(cols2):
                differences.append({
                    "position": [row, col],
                    "grid1_value": "N/A (row doesn't exist)",
                    "grid2_value": grid2[row][col]
                })
        
        # Extra columns in grid1 (if cols1 > cols2)
        for row in range(min_rows):
            for col in range(min_cols, cols1):
                differences.append({
                    "position": [row, col],
                    "grid1_value": grid1[row][col],
                    "grid2_value": "N/A (column doesn't exist)"
                })
        
        # Extra columns in grid2 (if cols2 > cols1)
        for row in range(min_rows):
            for col in range(min_cols, cols2):
                differences.append({
                    "position": [row, col],
                    "grid1_value": "N/A (column doesn't exist)",
                    "grid2_value": grid2[row][col]
                })
        
        total_cells = max(rows1 * cols1, rows2 * cols2)
        result = {
            "status": "success",
            "grid1_size": f"{rows1}x{cols1}",
            "grid2_size": f"{rows2}x{cols2}",
            "total_cells_compared": total_cells,
            "different_cells": len(differences),
            "match_percentage": round(100 * (1 - len(differences) / total_cells), 1) if total_cells > 0 else 0,
            "differences": differences
        }
        
        return result
    
    def call_ai_with_image(self, text_prompt: str, image_paths: List[str], image_descriptions: List[str] = None) -> str:
        """Call OpenAI with text and images with optional descriptions"""

        print(f"[START: {self.current_task_name}]")
        print("\n" + "="*80)
        print(f"PHASE PROMPT TO OPENAI:")
        print("-"*80)
        print(text_prompt)
        print("="*80)
        print(f"[END: {self.current_task_name}]")

        
        # Prepare content with images for the new responses API
        content = [{"type": "input_text", "text": text_prompt}]
        
        for i, image_path in enumerate(image_paths):
            base64_image = self.encode_image(image_path)

            if not image_descriptions or i >= len(image_descriptions):
                print("ERROR: image_descriptions must be provided with images")

            content.append({
                "type": "input_text",
                "text": f"{image_descriptions[i]}:"
            })

            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}"
            })
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": content})
        
        # Create the API call with tools always enabled
        call_params = {
            "model": "gpt-5",
            "reasoning": {
                "effort": "high"
                # "max_tokens": 10 * 1000
            },
            "input": self.conversation_history,
            "tools": [VISUALIZATION_TOOL, DIFF_TOOL],
            "tool_choice": "auto"
        }
        
        # Keep calling the API while tool calls are being made
        max_iterations = 20
        iteration = 0
        any_tools_used = False
        
        # Make first API call before the loop
        print(f"\n📡 Initial API Call")
        response = self.client_openai.responses.create(**call_params)
        
        # Log the response structure
        print(f"📦 Response output contains {len(response.output)} items")
        
        # Add response output to conversation history
        if response.output:
            self.conversation_history.extend(response.output)
        
        # Main processing loop
        while iteration < max_iterations:
            # Log each output item if present
            if response.output:
                for idx, item in enumerate(response.output):
                    print(f"  Item {idx}: type={item.type}")
                    if item.type == "message":
                        # Try to extract text from message content
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'type'):
                                    print(f"    Content type: {content_item.type}")
                                    if content_item.type == "output_text" and hasattr(content_item, 'text'):
                                        print(f"[START: {self.current_task_name}]")
                                        preview = content_item.text[:1500] + "..." if len(content_item.text) > 1500 else content_item.text
                                        print(f"RESPONSE (Length: {len(content_item.text)} characters):")
                                        print(f"{preview}")
                                        print(f"[END: {self.current_task_name}]")
            
            # Check for function calls in the output
            has_function_call = False
            for item in response.output:
                if item.type == "function_call":
                    has_function_call = True
                    any_tools_used = True  # Track that we've used tools
                    print(f"\n🔧 Function call detected: {item.name if hasattr(item, 'name') else 'unknown'}")
                    
                    # Parse the function arguments
                    args = json.loads(item.arguments)
                    
                    if item.name == "visualize_grid":
                        # Create visualization
                        grid = args["grid"]
                        print(f"  Creating visualization for grid of size {len(grid)}x{len(grid[0]) if grid else 0}")
                        img_path = self.create_grid_image(grid, label="tool")
                        base64_img = self.encode_image(img_path)
                        
                        # Add function result to conversation history
                        self.conversation_history.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({
                                "image_url": f"data:image/png;base64,{base64_img}",
                                "status": "success"
                            })
                        })
                        print(f"  ✅ Visualization created and added to conversation")
                    
                    elif item.name == "diff_grids":
                        # Compare two grids numerically
                        grid1 = args["grid1"]
                        grid2 = args["grid2"]
                        
                        result = self.diff_grids(grid1, grid2)
                        
                        print(f"  📊 Diff completed: {result['different_cells']} differences found ({result['match_percentage']}% match)")
                        
                        # Add function result to conversation history
                        self.conversation_history.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(result)
                        })
                        print(f"  ✅ Diff result added to conversation")
                    
                    iteration += 1
            
            # Also try to extract any text response even if there were tool calls
            if hasattr(response, 'output_text') and response.output_text:
                print(f"[START: {self.current_task_name}]")
                print(f"\n💬 Response text: {response.output_text[:500]}..." if len(response.output_text) > 500 else f"\n💬 Response text: {response.output_text}")
                print(f"[END: {self.current_task_name}]")
            
            if not any_tools_used: 
                break
            else:
                print("\n📡 Making final API call after tool use...")
                call_params["input"] = self.conversation_history
                response = self.client_openai.responses.create(**call_params)
                if response.output:
                    self.conversation_history.extend(response.output)
                if hasattr(response, 'output_text') and response.output_text:
                    print(f"[START: {self.current_task_name}]")
                    print(f"\nFinal response text: {response.output_text[:500]}..." if len(response.output_text) > 500 else f"\nFinal response text: {response.output_text}")
                    print(f"[END: {self.current_task_name}]")
                else:
                    print("🚨🚨🚨 CRITICAL WARNING: No response text available! This will likely cause failures! 🚨🚨🚨")

                # Check if this response has function calls
                response_has_function_calls = False
                if response.output:
                    for item in response.output:
                        if item.type == "function_call":
                            response_has_function_calls = True
                            break

                if not response_has_function_calls:
                    break
        
        if iteration >= max_iterations:
            print("\n⚠️ Warning: Maximum iterations reached")
        
        # Debug output
        print(f"[START: {self.current_task_name}]")
        print("\n" + "="*80)
        print(f"Images included: {len(image_paths)}")
        print(f"Tool call iterations made: {iteration}")
        print("-"*80)
        
        # Print response or warning
        if response and hasattr(response, 'output_text') and response.output_text:
            print(f"RESPONSE (Length: {len(response.output_text)} characters):")
            print(response.output_text[:5000] + "..." if len(response.output_text) > 5000 else response.output_text)
        else:
            print("🚨🚨🚨 CRITICAL WARNING: No response text available! This will likely cause failures! 🚨🚨🚨")
        
        print("="*80)
        print(f"[END: {self.current_task_name}]")
        
        return response.output_text if response and hasattr(response, 'output_text') else ""
    
    def parse_grid_from_response(self, response: str) -> Optional[List[List[int]]]:
        """Parse a grid from the AI's response"""
        import re
        
        # Find all bracket-enclosed sequences
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, response)
        
        grid = []
        for match in matches:
            # Remove brackets and parse numbers
            numbers = match.strip('[]').replace(' ', '').split(',')
            row = [int(n) for n in numbers if n]
            if row:
                grid.append(row)
        
        return grid if grid else None
    
    def solve(self, task_file: str) -> Tuple[bool, Optional[List[List[int]]], int]:
        """
        Main solving loop using two-prompt approach
        Returns (success, predicted_output, num_phases)
        """
        # Extract task name from file path
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        
        # Load the task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Task contains {len(task['train'])} training examples and {len(task['test'])} test examples")
        
        # Reset conversation history
        self.conversation_history = []
        num_phases = 0
        
        # Phase 1: Create all images and send with descriptions
        print("\n" + "="*80)
        print("=== Phase 1: Sending all images with descriptions ===")
        print("="*80)
        
        # Create all training example images
        all_images = []
        image_descriptions = []
        
        for i, example in enumerate(task['train']):
            input_img = self.create_grid_image(example['input'], label=f"train{i+1}_input")
            output_img = self.create_grid_image(example['output'], label=f"train{i+1}_output")
            all_images.extend([input_img, output_img])
            image_descriptions.append(f"Training Example {i+1} Input")
            image_descriptions.append(f"Training Example {i+1} Output")
        
        # Create test input image
        test_input_img = self.create_grid_image(task['test'][0]['input'], label="test_input")
        all_images.append(test_input_img)
        image_descriptions.append("Test Input")
        
        # Also save test output for logging (but don't send it)
        test_output_img = self.create_grid_image(task['test'][0]['output'], label="test_output")
        print(f"  Test output image saved to: {test_output_img}")
        
        prompt_1 = f"""
You are looking at a visual puzzle. I'll show you examples of inputs and their corresponding outputs.

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significance; properties of the symbols may convey this semantic significance. You need to find what properties carry semantic significance and what properties do not contribute to decision making. 

Some rules have to be applied based on context. Do not fixate on superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them. 

See if you can make sense of the puzzle, then I will provide the actual data.

"""


        prompt_2 = """
Now I'll show you the data that generated these images. Refine the patterns you observed earlier to help you solve the test problem.

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significance; properties of the symbols may convey this semantic significance. You need to find what properties carry semantic significance and what properties do not contribute to decision making. 

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense. You can try using a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool. 

Some rules have to be applied based on context. Do not fixate on superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them. 

In the test input, the game may be set up in a novel orientation; elements may be rotated or in places they have never appeared before; the color scheme may also be novel. Your job is to apply the rules you've developed to a novel situation that you have no input/output for. 

The tools here are going to be your biggest asset. In the past, you were able to find the correct solution more often when you leveraged the tools heavily. Continue iterating until the tool generates the correct outputs in all training examples.

Based on the consistent pattern you've identified from all the training examples, generate a possible output grid. 

You can use a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool. 

Once you feel that you've identified the rules, STOP. Further iteration after identifying a good set of rules may lead to confusion. Once you have a solution that you are confident in, output the result.

IMPORTANT: Regardless of how much iteration you do, *always* run your final prediction through the 'visualize_grid' tool and make sure it matches your expectations before returning an output. If the output does not match your expectation, continue refining your approach.

Produce an explanation of what the output should look like and observations of the output that you generated with 'visualize_grid'. Explain why parts of the generated image make sense or don't.
"""

        prompt_3 = """
Think about whether your last guess makes sense and refine your approach if necessary. You can continue iterating with tools if you think that would be helpful.

Generate your final guess for the output of the test puzzle.

IMPORTANT: Provide your answer as a grid in the exact same format, with square brackets and comma-separated values. Make sure the dimensions are correct.
"""

        for i, example in enumerate(task['train']):
            prompt_2 += f"Training Example {i+1} Input: \n"
            prompt_2 += self.format_grid(task['train'][i]['input']) + "\n\n"

            prompt_2 += f"Training Example {i+1} Output: \n"
            prompt_2 += self.format_grid(task['train'][i]['output']) + "\n\n"

        prompt_2 += f"Test input: \n"
        prompt_2 += self.format_grid(task['test'][0]['input'])

        response_1 = self.call_ai_with_image(prompt_1, all_images, image_descriptions)
        num_phases += 1

        response_2 = self.call_ai_with_image(prompt_2, [], [])
        num_phases += 1

        response_3 = self.call_ai_with_image(prompt_3, [], [])
        num_phases += 1
        
        # Parse the predicted output
        predicted_output = self.parse_grid_from_response(response_3)
        
        # Check if we got a valid prediction
        if not predicted_output:
            print("\n❌ Could not parse a valid grid from the response")
            return False, None, num_phases
        
        # Compare with actual test output (if available)
        if 'output' in task['test'][0] and task['test'][0]['output']:
            actual_output = task['test'][0]['output']
            
            if predicted_output == actual_output:
                print("\n✅ SUCCESS! Predicted output matches actual output!")
                return True, predicted_output, num_phases
            else:
                print("\n❌ Predicted output does not match actual output")
                print(f"Predicted: {predicted_output[:3]}..." if len(predicted_output) > 3 else predicted_output)
                print(f"Actual: {actual_output[:3]}..." if len(actual_output) > 3 else actual_output)
                return False, predicted_output, num_phases
        else:
            print("\n⚠️ No test output available for comparison")
            print(f"Generated prediction: {predicted_output[:3]}..." if len(predicted_output) > 3 else predicted_output)
            return False, predicted_output, num_phases


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc_visual_solver.py <task_json_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        solver = ARCVisualSolver()
        success, prediction, num_phases = solver.solve(task_file)
        
        print(f"\n{'='*80}")
        print(f"Solving complete!")
        print(f"Phases used: {num_phases}")
        print(f"Result ({solver.current_task_name}): {'SUCCESS ✅' if success else 'FAILED ❌'}")
        if prediction:
            # Save prediction as image using our method
            pred_path = solver.create_grid_image(prediction, label="final_prediction")
            print(f"Prediction saved to: {pred_path}")
        print(f"{'='*80}")
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()