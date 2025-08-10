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
    "function": {
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
}


class ARCVisualSolver:
    def __init__(self):
        """Initialize the ARC visual solver with API credentials"""
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        if not self.api_key_openai:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self.client_openai = OpenAI(api_key=self.api_key_openai)
        
        self.conversation_history = []
        
        # Use img_tmp directory in project root
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_tmp")
    
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from a JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display in the prompt"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def create_grid_image(self, grid: List[List[int]], cell_size: int = 30) -> str:
        """Create an image from a grid and return the file path"""
        img = grid_to_image(grid, cell_size)
        # Save to temp file
        temp_path = os.path.join(self.temp_dir, f"grid_{len(os.listdir(self.temp_dir))}.png")
        img.save(temp_path)
        return temp_path
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle a tool call from the AI"""
        if tool_call.function.name == "visualize_grid":
            args = json.loads(tool_call.function.arguments)
            grid = args["grid"]
            
            # Create visualization
            img_path = self.create_grid_image(grid)
            
            # Encode image to base64
            base64_img = self.encode_image(img_path)
            
            # Return the image in the expected format
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }
                ]
            }
        return None
    
    def call_ai_with_image(self, text_prompt: str, image_paths: List[str]) -> str:
        """Call OpenAI with text and images"""
        
        # Prepare content with images for OpenAI
        content = [{"type": "text", "text": text_prompt}]
        
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        
        messages = self.conversation_history + [{"role": "user", "content": content}]
        
        # Create the API call with tools always enabled
        call_params = {
            model="gpt-5-thinking",
            reasoning={"effort": "high"},
            "messages": messages,
            "tools": [VISUALIZATION_TOOL],
            "tool_choice": "auto"
        }
        
        # Keep calling the API while tool calls are being made
        max_iterations = 10
        iteration = 0
        final_message = None
        
        while iteration < max_iterations:
            response = self.client_openai.chat.completions.create(**call_params)
            assistant_message = response.choices[0].message
            
            # Add assistant message to messages for context
            call_params["messages"].append(assistant_message)
            
            # Handle tool calls if present
            if assistant_message.tool_calls:
                # Process tool calls
                for tool_call in assistant_message.tool_calls:
                    result = self.handle_tool_call(tool_call)
                    if result:
                        call_params["messages"].append(result)
                        print(f"\n🔧 Tool called: {tool_call.function.name}")
                iteration += 1
                # Continue loop to allow more tool calls
            else:
                # No more tool calls, we have the final response
                final_message = assistant_message.content
                break
        
        if final_message is None:
            print("\n⚠️ Warning: Maximum iterations reached")
            final_message = call_params["messages"][-1].content if call_params["messages"] else "No response"
        
        # Add to conversation history (simplified version without images for history)
        self.conversation_history.append({"role": "user", "content": text_prompt})
        self.conversation_history.append({"role": "assistant", "content": final_message})
        
        # Debug output
        print("\n" + "="*80)
        print(f"PHASE PROMPT TO OPENAI:")
        print("-"*80)
        print(text_prompt)
        print(f"Images included: {len(image_paths)}")
        print(f"Tool call iterations: {iteration}")
        print("-"*80)
        print(f"RESPONSE FROM OPENAI:")
        print("-"*80)
        print(final_message)
        print("="*80)
        
        return final_message
    
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
        Main solving loop using phased visual approach
        Returns (success, predicted_output, num_phases)
        """
        # Load the task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Task contains {len(task['train'])} training examples and {len(task['test'])} test examples")
        
        # Reset conversation history
        self.conversation_history = []
        num_phases = 0
        
        # Phase 1: Show first training example with visual
        print("\n" + "="*80)
        print("=== Phase 1: First training example ===")
        print("="*80)
        
        input_img_1 = self.create_grid_image(task['train'][0]['input'])
        output_img_1 = self.create_grid_image(task['train'][0]['output'])
        
        prompt_1 = f"""
You are looking at a visual puzzle. I'll show you examples of inputs and their corresponding outputs.

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significants; properties of the symbols may convey this semantic significants. You need to find what properties carry semantic significance and what properties do not contribute to decision making. 

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense. You can try using a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool. 

Some rules have to be applied based on context. Do not fixate of superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them. 

Here's the first training example:

Input grid:
{self.format_grid(task['train'][0]['input'])}

Output grid:
{self.format_grid(task['train'][0]['output'])}

"""

        response_1 = self.call_ai_with_image(prompt_1, [input_img_1, output_img_1])
        num_phases += 1
        
        # Phase 2: Show second training input, ask for prediction
        if len(task['train']) > 1:
            print("\n" + "="*80)
            print("=== Phase 2: Second training input - predict output ===")
            print("="*80)
            
            input_img_2 = self.create_grid_image(task['train'][1]['input'])
            
            prompt_2 = f"""
Now I'll show you the second training input. Based on the pattern you observed in the first example, try to predict what the output should be.

Second training input:
{self.format_grid(task['train'][1]['input'])}
"""

            response_2 = self.call_ai_with_image(prompt_2, [input_img_2])
            num_phases += 1
            
            # Phase 3: Show actual second training output
            print("\n" + "="*80)
            print("=== Phase 3: Actual second training output ===")
            print("="*80)
            
            output_img_2 = self.create_grid_image(task['train'][1]['output'])
            
            prompt_3 = f"""Here's the actual output for the second training example:

Output grid:
{self.format_grid(task['train'][1]['output'])}
"""

            response_3 = self.call_ai_with_image(prompt_3, [output_img_2])
            num_phases += 1
        
        # If there are more training examples, show them
        for i in range(2, len(task['train'])):
            print(f"\n" + "="*80)
            print(f"=== Additional training example {i+1} ===")
            print("="*80)
            
            input_img = self.create_grid_image(task['train'][i]['input'])
            output_img = self.create_grid_image(task['train'][i]['output'])
            
            prompt = f"""Here's training example {i+1}:

Input:
{self.format_grid(task['train'][i]['input'])}

Output:
{self.format_grid(task['train'][i]['output'])}
"""

            response = self.call_ai_with_image(prompt, [input_img, output_img])
            num_phases += 1
        
        # Phase 4: Test input - ask for output
        print("\n" + "="*80)
        print("=== Phase 4: Test input - generate output ===")
        print("="*80)
        
        test_input_img = self.create_grid_image(task['test'][0]['input'])
        
        prompt_test = f"""Now, here's the test input. Apply the pattern you've learned to generate the output.

Test input:
{self.format_grid(task['test'][0]['input'])}

Look at the visual representation below. Based on the consistent pattern you've identified from all the training examples, generate the output grid. 

You can use a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool. 

IMPORTANT: Provide your answer as a grid in the exact same format, with square brackets and comma-separated values. Make sure the dimensions are correct."""

        response_test = self.call_ai_with_image(prompt_test, [test_input_img])
        num_phases += 1
        
        # Parse the predicted output
        predicted_output = self.parse_grid_from_response(response_test)
        
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
        print(f"Result: {'SUCCESS ✅' if success else 'FAILED ❌'}")
        if prediction:
            # Save prediction as image
            pred_img = grid_to_image(prediction)
            pred_img.save("img_tmp/prediction.png")
            print(f"Prediction saved to: img_tmp/prediction.png")
        print(f"{'='*80}")
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()