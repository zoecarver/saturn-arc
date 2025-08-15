#!/usr/bin/env python3
"""
ARC-AGI Streaming DSL Solver
Uses GPT-5 with streaming to solve ARC puzzles through phased DSL generation
"""

import json
import os
import sys
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Generator, Tuple
from enum import Enum
from openai import OpenAI
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path

from arc_visualizer import grid_to_image, ARC_COLORS

# Expected Flow:
# 
# 1. train_0 (First training example)
#    - Prompt: TRAIN_0_PROMPT
#    - Expected output: DSL program
# 
# 2. train_1_predict
#    - Prompt: TEST_PROMPT
#    - Expected output: Grid of numbers
# 
# 3. train_1_verify
#    - Prompt: TRAIN_VERIFY_PROMPT
#    - Expected output: DSL program
# 
# 4. train_2_predict
#    - Prompt: TEST_PROMPT
#    - Expected output: Grid of numbers
# 
# 5. train_2_verify
#    - Prompt: TRAIN_VERIFY_PROMPT
#    - Expected output: DSL program
# 
# 6. test_refine
#    - Prompt: TEST_REFINE_PROMPT
#    - Expected output: DSL program tailored for test input
# 
# 7. test_predict
#    - Prompt: TEST_PROMPT
#    - Expected output: Grid of numbers


# API Configuration
MODEL_NAME = "gpt-5"
REASONING_EFFORT = "high"
MAX_ITERATIONS = 20
CELL_SIZE = 30

# DSL Examples
DSL_EXAMPLES = """
This DSL is structured natural language - you can invent functions, operators, and syntax as needed to clearly express your intent. The goal is to convey meaning in a structured, concise way rather than follow rigid syntax rules. Think of it as pseudocode that prioritizes clarity: if you need to "find all red squares that touch blue circles," just write Find("red squares", where="adjacent to blue circles"). Create helper functions like Lambda("holes", "black regions fully enclosed by shape") when concepts repeat. The DSL should read like a clear recipe for the transformation, using intuitive names and logical flow. Focus on expressing the algorithm's logic rather than worrying about exact function signatures.

IMPORTANT: Avoid traditional program synthesis with complex control flow. Prefer natural language descriptions over loops and conditionals.

GOOD Example (Natural Language):
Lambda("fill_between_colors", "Find zero runs bounded by same color on both ends, fill with that color")
Apply("fill_between_colors", to="each row and column")

BAD Example (Complex Control Flow):
Lambda("fill", {{
    for row in grid:
        for i in range(...):
            if row[i] == 0 and ...:
                fill(row[i:j], color)
}})

Instead of nested loops and if statements, describe WHAT should happen, not HOW to iterate through it.

Example Patterns
1. Color Mapping from Palette
dslpalette = Find("palette", where="separated by divider")
color_map = {{}}
for swatch in palette {{
    holes = Count("black squares", within=swatch)
    color = GetColor(swatch, exclude="black")
    color_map[holes] = color
}}
2. Shape Processing with Hole Detection
dslshapes = Find("connected components", where="color is orange")
for shape in shapes {{
    holes = Count(Find("black regions", within=shape, where="fully enclosed"))
    if holes in color_map {{
        Paint(shape, color_map[holes], preserve="holes as black")
    }}
}}
3. Path Finding and Recoloring
dslterminals = Find("connected_components", where="color == terminal_color")
bridge = Filter(traversable_components, where={{
    Intersects(component, adjacent_to_terminals[0]) AND 
    Intersects(component, adjacent_to_terminals[1])
}})
Paint(bridge, new_color)
4. Stroke Drawing from Instructions
dslstrokes = {{
    1: Lambda("right_3", {{ Draw(canvas, [row], [col..col+2]); col+=2; row+=1 }}),
    2: Lambda("left_2", {{ Draw(canvas, [row], [col-1..col]); col-=1; row+=1 }})
}}
for instruction in sequence {{
    Execute(strokes[instruction])
}}
5. Palette-Guided Line Completion
program {{
    grid = Input()
    palette = Find("2x N block", where="top edge, rightmost, nonzero")
    color_map = BuildMapping(palette, toprow="source", botrow="target")
 
    FillRuns = Lambda("fill zeros between matching colors") {{
        runs = Find("zero runs", within=line, where="bounded by key on both ends")
        Paint(runs, fill)
    }}
 
    for key, fill in color_map {{
        Apply(FillRuns, to="all rows and columns", args=[key, fill])
    }}
 
    Output(grid)
}}
"""

# Phase Prompts
TRAIN_0_PROMPT = """
You are looking at a visual puzzle. I'll show you examples of inputs and their corresponding outputs. Here is the first trianing pair.

Input:
{input_grid}

Output:
{output_grid}

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significance; properties of the symbols may convey this semantic significance. You need to find what properties carry semantic significance and what properties do not contribute to decision making.

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense. You can try using a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool.

Some rules have to be applied based on context. Do not fixate on superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them.

Generate ONLY the DSL program. Focus on:
- Spatial relationships and movements
- Color transformations
- Pattern detection and replication
- Conditional rules based on context

Output format: DSL program only, no explanations.

{dsl_examples}

IMPORTANT: Your response must be ONLY a DSL program. No grids, no numbers, no explanations - just the DSL code.
"""

TRAIN_VERIFY_PROMPT = """
The actual output is:

{output_grid}

If you did not produce the correct output earlier, refine your approach and use the tool to iterate.

Look at the visual representation of your submission and see if it makes sense. 

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significance; properties of the symbols may convey this semantic significance. You need to find what properties carry semantic significance and what properties do not contribute to decision making.

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense. You can try using a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool.

Some rules have to be applied based on context. Do not fixate on superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them.

Continue iterating until the tool generates the correct outputs in all training examples.

Refine your DSL program by:
1. Testing it mentally on ALL training inputs you've seen so far
2. Verify it produces the correct output for training example 1
3. Verify it produces the correct output for training example 2
4. If there are more training examples, verify it works for those too
5. Identify any discrepancies and adjust the rules
6. Generate a unified DSL program that successfully transforms ALL training inputs to their correct outputs

The DSL program MUST work for every single training example. Test it on each one before finalizing.
Output format: DSL program only.

{dsl_examples}

IMPORTANT: Your response must be ONLY a DSL program. No grids, no numbers, no explanations - just the DSL code.
"""

TEST_REFINE_PROMPT = """
Now look at the test input and refine your DSL program specifically for this case.

Test Input:
{input_grid}

Look at this test input visually and carefully. Consider:
- Are there any patterns or structures in this test input that differ from the training examples?
- Does the test input have unique characteristics that need special handling?
- How can you tailor your DSL to handle this specific test case while maintaining the general rules?

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significance; properties of the symbols may convey this semantic significance. You need to find what properties carry semantic significance and what properties do not contribute to decision making.

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense.

Some rules have to be applied based on context. Do not fixate on superficial patterns; find what properties have semantic significance and use those as context.

Generate a DSL program that:
1. Incorporates all the rules you've learned from training
2. Is specifically tailored to handle this test input
3. Accounts for any unique aspects of this test case

Output format: DSL program only.

{dsl_examples}

IMPORTANT: Your response must be ONLY a DSL program. No grids, no numbers, no explanations - just the DSL code.
"""

TEST_PROMPT = """
{instructions}

Input:
{input_grid}

IMPORTANT: Do NOT output a DSL program. Instead, EXECUTE your learned transformation rules mentally and output the ACTUAL RESULT as a grid of numbers.

Apply your learned transformations step by step:
1. Look at the test input grid
2. Apply the transformation rules you discovered from the training examples
3. Generate the actual output grid

Output ONLY the resulting grid as an array of arrays with square brackets and comma-separated values.
Each row should be on its own line for clarity.

Example of the required output format:
[[0, 1, 2, 3],
 [4, 5, 6, 7],
 [8, 9, 0, 1],
 [2, 3, 4, 5]]

Your output should be ONLY the grid, nothing else. No explanations, no DSL code, just the numerical grid.

IMPORTANT: Your response must be ONLY a grid of numbers in bracket format. Do NOT output any DSL code or explanations.
"""


class Phase(Enum):
    """Phases of the solving process"""
    TRAIN_N = "train_n"
    TRAIN_PREDICT = "train_predict"
    TRAIN_VERIFY = "train_verify"
    TEST = "test"


@dataclass
class StreamEvent:
    """Represents a streaming event from the API"""
    type: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskData:
    """Structured representation of an ARC task"""
    name: str
    train: List[Dict[str, List[List[int]]]]
    test: List[Dict[str, List[List[int]]]]
    
    @classmethod
    def from_file(cls, file_path: str) -> 'TaskData':
        """Load task from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        name = Path(file_path).stem
        return cls(name=name, train=data['train'], test=data['test'])


def format_grid(grid: List[List[int]]) -> str:
    """Format grid for display"""
    return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])


def parse_grid_from_response(response: str) -> Optional[List[List[int]]]:
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


def create_grid_image(grid: List[List[int]], task_name: str, label: str) -> str:
    """Create and save grid image"""
    img = grid_to_image(grid, cell_size=CELL_SIZE)
    img_dir = Path("img_tmp")
    img_dir.mkdir(exist_ok=True)
    path = img_dir / f"{task_name}_{label}.png"
    img.save(path)
    return str(path)


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


class APIClient:
    """Handles API interactions with OpenAI (non-streaming)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        self.client = OpenAI(api_key=self.api_key)
    
    def get_response(self, messages: List[Dict]) -> str:
        """Get response from the API and log reasoning"""
        response = self.client.responses.create(
            model=MODEL_NAME,
            reasoning={"effort": REASONING_EFFORT},
            input=messages
        )
        
        # Log response structure
        print(f"\n📦 Response output contains {len(response.output)} items")
        
        reasoning_text = ""
        message_text = ""
        
        # Process each output item
        for idx, item in enumerate(response.output):
            print(f"  Item {idx}: type={item.type}")
            
            if item.type == "reasoning":
                # Debug: print the full item to understand its structure
                print(f"    Full reasoning item: {item}")
                
                # Check if there's encrypted content
                if hasattr(item, 'encrypted_content') and item.encrypted_content:
                    print(f"    Encrypted content present (length: {len(str(item.encrypted_content))})")
                
                # Check if there's regular content
                if hasattr(item, 'content') and item.content:
                    print(f"    Content: {item.content}")
                
                # Extract and log reasoning summary
                if hasattr(item, 'summary') and item.summary and len(item.summary) > 0:
                    for summary_item in item.summary:
                        if hasattr(summary_item, 'type') and summary_item.type == "summary_text":
                            if hasattr(summary_item, 'text'):
                                reasoning_text = summary_item.text
                                print(f"\n💭 REASONING:")
                                print("-" * 60)
                                print(reasoning_text)
                                print("-" * 60)
                                break
                else:
                    print(f"    Note: Reasoning summary is empty. This might be intentional for this model/configuration.")
            
            elif item.type == "message":
                # Extract message text
                if hasattr(item, 'content'):
                    for content_item in item.content:
                        if hasattr(content_item, 'type'):
                            print(f"    Content type: {content_item.type}")
                            if content_item.type == "output_text" and hasattr(content_item, 'text'):
                                message_text = content_item.text
        
        # Also check for direct output_text
        if hasattr(response, 'output_text') and response.output_text:
            final_text = response.output_text
        else:
            final_text = message_text
        
        return final_text


class Logger:
    """Handles logging of thinking and outputs"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / f"{task_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.thinking_buffer = []
    
    def log_thinking(self, content: str):
        """Log and print thinking content incrementally"""
        print(f"💭 {content}")
        self.thinking_buffer.append(content)
        with open(self.log_file, 'a') as f:
            f.write(f"[THINKING] {content}\n")
    
    def log_message(self, content: str):
        """Log message content"""
        with open(self.log_file, 'a') as f:
            f.write(f"[MESSAGE] {content}\n")
    
    def log_phase(self, phase: str, content: str):
        """Log phase completion"""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"Phase: {phase}")
        print(separator)
        
        with open(self.log_file, 'a') as f:
            f.write(f"\n{separator}\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"{separator}\n")
            f.write(f"{content}\n")


def prepare_message_content(text: str, images: Optional[List[Tuple[str, str]]] = None) -> List[Dict]:
    """
    Prepare message content with text and labeled images
    images: List of (label, image_path) tuples
    """
    content = [{"type": "input_text", "text": text}]
    
    if images:
        for label, img_path in images:
            # Add label text
            content.append({"type": "input_text", "text": f"\n{label}:"})
            # Add image
            base64_img = encode_image(img_path)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_img}"
            })
    
    return content


def execute_training_phase(
    client: APIClient,
    task_data: TaskData,
    train_idx: int,
    logger: Logger,
    conversation_history: List[Dict],
    first_example: bool = False
) -> str:
    """Execute a training phase for a specific example"""
    # Create images
    input_img = create_grid_image(
        task_data.train[train_idx]['input'],
        task_data.name,
        f"train_{train_idx}_input"
    )
    output_img = create_grid_image(
        task_data.train[train_idx]['output'],
        task_data.name,
        f"train_{train_idx}_output"
    )
    
    # Prepare prompt
    prompt = TRAIN_0_PROMPT.format(
        index=train_idx + 1,
        input_grid=format_grid(task_data.train[train_idx]['input']),
        output_grid=format_grid(task_data.train[train_idx]['output']),
        dsl_examples=DSL_EXAMPLES
    )
    
    # Prepare message with labeled images
    images = [
        (f"Training {train_idx + 1} Input", input_img),
        (f"Training {train_idx + 1} Output", output_img)
    ]
    
    content = prepare_message_content(prompt, images)
    conversation_history.append({"role": "user", "content": content})
    
    # Get response
    dsl = client.get_response(conversation_history)
    logger.log_message(dsl)
    print(f"\n📝 Output:\n{dsl}")
    conversation_history.append({"role": "assistant", "content": dsl})
    
    logger.log_phase(f"train_{train_idx}", dsl)
    return dsl


def execute_prediction_phase(
    client: APIClient,
    task_data: TaskData,
    idx: int,
    logger: Logger,
    conversation_history: List[Dict],
    is_test: bool,
    is_validation: bool
) -> Tuple[str, Optional[List[List[int]]]]:
    """Execute a prediction phase"""
    if is_test:
        grid = task_data.test[0]['input']
        label = "test_input"
        instructions = "Now it's time to apply what you've learned to generate the actual output for this test input."
        prompt = TEST_PROMPT.format(
            instructions=instructions,
            input_grid=format_grid(grid)
        )
        image_label = "Test Input"
        
        # Save test output for debugging (if available), but don't send it
        if 'output' in task_data.test[0] and task_data.test[0]['output']:
            test_output_img = create_grid_image(
                task_data.test[0]['output'],
                task_data.name,
                "test_output"
            )
            print(f"    [Debug] Test output saved: {test_output_img}")
    else:
        grid = task_data.train[idx]['input']
        label = f"train_{idx}_input"
        instructions = f"Based on the patterns you've identified, predict the output for Training Example {idx + 1}."
        prompt = TEST_PROMPT.format(
            instructions=instructions,
            input_grid=format_grid(grid)
        )
        image_label = f"Training {idx + 1} Input (Prediction)"
    
    # Create image
    input_img = create_grid_image(grid, task_data.name, label)
    
    # Prepare message
    images = [(image_label, input_img)]
    content = prepare_message_content(prompt, images)
    conversation_history.append({"role": "user", "content": content})
    
    # Get response
    dsl = client.get_response(conversation_history)
    logger.log_message(dsl)
    print(f"\n📝 Output:\n{dsl}")
    conversation_history.append({"role": "assistant", "content": dsl})
    
    phase_name = "test_predict" if is_test else f"train_{idx}_predict"
    logger.log_phase(phase_name, dsl)
    
    # Parse the grid output when needed for validation or test
    predicted_grid = None
    if is_test or is_validation:
        predicted_grid = parse_grid_from_response(dsl)
    
    return dsl, predicted_grid


def execute_verification_phase(
    client: APIClient,
    task_data: TaskData,
    idx: int,
    logger: Logger,
    conversation_history: List[Dict],
    predicted_grid: Optional[List[List[int]]] = None
) -> str:
    """Execute a verification phase"""
    output_img = create_grid_image(
        task_data.train[idx]['output'],
        task_data.name,
        f"train_{idx}_output"
    )
    
    prompt = TRAIN_VERIFY_PROMPT.format(
        output_grid=format_grid(task_data.train[idx]['output']),
        dsl_examples=DSL_EXAMPLES
    )
    
    # Prepare message with both prediction and actual output
    images = []
    if predicted_grid:
        pred_img = create_grid_image(predicted_grid, task_data.name, f"train_{idx}_prediction")
        images.append(("This is a visual representation of what you submitted", pred_img))
    images.append((f"Training {idx + 1} Actual Output", output_img))
    
    content = prepare_message_content(prompt, images)
    conversation_history.append({"role": "user", "content": content})
    
    # Get response
    dsl = client.get_response(conversation_history)
    logger.log_message(dsl)
    print(f"\n📝 Output:\n{dsl}")
    conversation_history.append({"role": "assistant", "content": dsl})
    
    logger.log_phase(f"train_{idx}_verify", dsl)
    return dsl


class ARCSolver:
    """Main solver orchestrating the streaming DSL generation"""
    
    def __init__(self):
        self.client = APIClient()
    
    def solve(self, task_file: str) -> Tuple[bool, Optional[List[List[int]]], int]:
        """
        Solve an ARC task using streaming DSL generation
        Returns: (success, predicted_output, num_phases)
        """
        task_data = TaskData.from_file(task_file)
        logger = Logger(task_data.name)
        conversation_history = []
        
        print(f"\nSolving task: {task_data.name}")
        print(f"Training examples: {len(task_data.train)}, Test examples: {len(task_data.test)}")
        
        num_phases = 0
        final_dsl = None
        
        # Process first training example
        dsl = execute_training_phase(
            self.client, task_data, 0, logger, conversation_history, first_example=True
        )
        final_dsl = dsl
        num_phases += 1
        
        # Process remaining training examples with predict-verify pattern
        for i in range(1, len(task_data.train)):
            # Predict phase
            _, predicted_grid = execute_prediction_phase(
                self.client, task_data, i, logger, conversation_history,
                is_test=False, is_validation=True
            )
            num_phases += 1
            
            # Verify phase
            dsl = execute_verification_phase(
                self.client, task_data, i, logger, conversation_history, predicted_grid
            )
            final_dsl = dsl
            num_phases += 1
        
        # Test refinement phase - show test input and ask for refined DSL
        print("\n" + "="*60)
        print("Test refinement - tailoring DSL for test input")
        print("="*60)
        
        test_input_img = create_grid_image(
            task_data.test[0]['input'],
            task_data.name,
            "test_input_refine"
        )
        
        refine_prompt = TEST_REFINE_PROMPT.format(
            input_grid=format_grid(task_data.test[0]['input']),
            dsl_examples=DSL_EXAMPLES
        )
        
        # Prepare message with test input image
        refine_images = [("Test Input", test_input_img)]
        refine_content = prepare_message_content(refine_prompt, refine_images)
        conversation_history.append({"role": "user", "content": refine_content})
        
        # Get refined DSL
        refined_dsl = self.client.get_response(conversation_history)
        logger.log_message(refined_dsl)
        print(f"\n📝 Refined DSL for test:\n{refined_dsl}")
        conversation_history.append({"role": "assistant", "content": refined_dsl})
        logger.log_phase("test_refine", refined_dsl)
        num_phases += 1
        
        # Test prediction
        _, predicted_output = execute_prediction_phase(
            self.client, task_data, 0, logger, conversation_history,
            is_test=True, is_validation=False
        )
        num_phases += 1
        
        # Validation
        success = False
        if predicted_output:
            if 'output' in task_data.test[0] and task_data.test[0]['output']:
                actual_output = task_data.test[0]['output']
                if predicted_output == actual_output:
                    print("\n✅ SUCCESS! Predicted output matches actual output!")
                    success = True
                else:
                    print("\n❌ Predicted output does not match actual output")
                    print(f"Predicted: {predicted_output[:3]}..." if len(predicted_output) > 3 else f"Predicted: {predicted_output}")
                    print(f"Actual: {actual_output[:3]}..." if len(actual_output) > 3 else f"Actual: {actual_output}")
            else:
                print("\n⚠️ No test output available for comparison")
                print(f"Generated prediction: {predicted_output[:3]}..." if len(predicted_output) > 3 else f"Generated prediction: {predicted_output}")
        else:
            print("\n❌ Could not parse a valid grid from the response")
        print(f"\n{'='*60}")
        print(f"Solving complete for {task_data.name}")
        print(f"Phases executed: {num_phases}")
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"{'='*60}")
        
        return success, predicted_output, num_phases


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc_streaming_solver.py <task_json_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        solver = ARCSolver()
        success, predicted_output, num_phases = solver.solve(task_file)
        
        if predicted_output:
            # Save prediction as image
            pred_img = create_grid_image(predicted_output, Path(task_file).stem, "final_prediction")
            print(f"\nPrediction image saved to: {pred_img}")
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()