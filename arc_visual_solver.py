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
    
    def call_ai_with_image(self, text_prompt: str, image_paths: List[str]) -> str:
        """Call OpenAI with text and images"""
        
        # Prepare content with images for the new responses API
        content = [{"type": "input_text", "text": text_prompt}]
        
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}"
            })
        
        messages = self.conversation_history + [{"role": "user", "content": content}]
        
        # Create the API call with tools always enabled
        call_params = {
            "model": "gpt-5",
            "reasoning": {
                "effort": "high"
                # "max_tokens": 10 * 1000
            },
            "input": messages,  # Use 'input' instead of 'messages' for responses API
            "tools": [VISUALIZATION_TOOL],
            "tool_choice": "auto"
        }
        
        # Keep calling the API while tool calls are being made
        max_iterations = 20
        iteration = 0
        final_message = None
        
        while iteration < max_iterations:
            print(f"\n📡 API Call iteration {iteration + 1}")
            response = self.client_openai.responses.create(**call_params)
            
            # Log the response structure
            print(f"📦 Response output contains {len(response.output)} items")
            
            # Add response output to input for context
            if response.output:
                call_params["input"] += response.output
                
                # Log each output item
                for idx, item in enumerate(response.output):
                    print(f"  Item {idx}: type={item.type}")
                    if item.type == "message":
                        # Try to extract text from message content
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'type'):
                                    print(f"    Content type: {content_item.type}")
                                    if content_item.type == "output_text" and hasattr(content_item, 'text'):
                                        preview = content_item.text[:200] + "..." if len(content_item.text) > 200 else content_item.text
                                        print(f"    Text preview: {preview}")
            
            # Check for function calls in the output
            has_function_call = False
            for item in response.output:
                if item.type == "function_call":
                    has_function_call = True
                    print(f"\n🔧 Function call detected: {item.name if hasattr(item, 'name') else 'unknown'}")
                    
                    # Parse the function arguments
                    args = json.loads(item.arguments)
                    
                    if item.name == "visualize_grid":
                        # Create visualization
                        grid = args["grid"]
                        print(f"  Creating visualization for grid of size {len(grid)}x{len(grid[0]) if grid else 0}")
                        img_path = self.create_grid_image(grid, label="tool")
                        base64_img = self.encode_image(img_path)
                        
                        # Add function result to input
                        call_params["input"].append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({
                                "image_url": f"data:image/png;base64,{base64_img}",
                                "status": "success"
                            })
                        })
                        print(f"  ✅ Visualization created and added to conversation")
                    
                    iteration += 1
            
            # Also try to extract any text response even if there were tool calls
            if hasattr(response, 'output_text') and response.output_text:
                print(f"\n💬 Response text: {response.output_text[:1500]}..." if len(response.output_text) > 500 else f"\n💬 Response text: {response.output_text}")
            
            if not has_function_call:
                print("\n✋ No more function calls, ending iteration")
                # Extract the final text response
                final_message = response.output_text if hasattr(response, 'output_text') else ""
                break
        
        if final_message is None:
            print("\n⚠️ Warning: Maximum iterations reached")
            # Try to extract text from the last item in input
            if call_params["input"]:
                last_item = call_params["input"][-1]
                if isinstance(last_item, dict) and "content" in last_item:
                    final_message = str(last_item["content"])
                else:
                    final_message = str(last_item)
            else:
                final_message = "No response"
        
        # Add to conversation history (simplified version without images for history)
        self.conversation_history.append({"role": "user", "content": text_prompt})
        self.conversation_history.append({"role": "assistant", "content": final_message})
        
        # Debug output
        print(f"[START: {self.current_task_name}]")
        print("\n" + "="*80)
        print(f"PHASE PROMPT TO OPENAI:")
        print("-"*80)
        print(text_prompt)
        print(f"Images included: {len(image_paths)}")
        print(f"Tool call iterations made: {iteration}")
        print("-"*80)
        print(f"FINAL RESPONSE FROM OPENAI:")
        if final_message:
            print(f"(Length: {len(final_message)} characters)")
        print("-"*80)
        print(final_message if final_message else "[No final message received]")
        print("="*80)
        print(f"[END: {self.current_task_name}]")
        
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
        # Extract task name from file path
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        
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
        
        input_img_1 = self.create_grid_image(task['train'][0]['input'], label="train1_input")
        output_img_1 = self.create_grid_image(task['train'][0]['output'], label="train1_output")
        
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
            
            input_img_2 = self.create_grid_image(task['train'][1]['input'], label="train2_input")
            
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
            
            output_img_2 = self.create_grid_image(task['train'][1]['output'], label="train2_output")
            
            prompt_3 = f"""Here's the actual output for the second training example:

Output grid:
{self.format_grid(task['train'][1]['output'])}

If you did not produce the correct output earlier, refine your approach and use the tool to iterate. 

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significants; properties of the symbols may convey this semantic significants. You need to find what properties carry semantic significance and what properties do not contribute to decision making. 

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense. You can try using a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool. 

Some rules have to be applied based on context. Do not fixate of superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them. 

Continue iterating until the tool generates the correct outputs in both training examples.
"""

            response_3 = self.call_ai_with_image(prompt_3, [output_img_2])
            num_phases += 1
        
        # If there are more training examples, show them
        for i in range(2, len(task['train'])):
            print(f"\n" + "="*80)
            print(f"=== Additional training example {i+1} ===")
            print("="*80)
            
            input_img = self.create_grid_image(task['train'][i]['input'], label=f"train{i+1}_input")
            output_img = self.create_grid_image(task['train'][i]['output'], label=f"train{i+1}_output")
            
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
        
        test_input_img = self.create_grid_image(task['test'][0]['input'], label="test_input")
        
        test_output_img = self.create_grid_image(task['test'][0]['output'], label="test_output")
        print(f"  Test output image saved to: {test_output_img}")
        
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