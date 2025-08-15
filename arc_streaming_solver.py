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

# API Configuration
MODEL_NAME = "gpt-5"
REASONING_EFFORT = "high"
MAX_ITERATIONS = 20
CELL_SIZE = 30

# Phase Prompts
TRAIN_N_PROMPT = """
You are looking at a visual puzzle. I'll show you examples of inputs and their corresponding outputs.

Training example {index}:

Input:
{input_grid}

Output:
{output_grid}

{instruction}

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

This DSL is structured natural language - you can invent functions, operators, and syntax as needed to clearly express your intent. The goal is to convey meaning in a structured, concise way rather than follow rigid syntax rules. Think of it as pseudocode that prioritizes clarity: if you need to "find all red squares that touch blue circles," just write Find("red squares", where="adjacent to blue circles"). Create helper functions like Lambda("holes", "black regions fully enclosed by shape") when concepts repeat. The DSL should read like a clear recipe for the transformation, using intuitive names and logical flow. Focus on expressing the algorithm's logic rather than worrying about exact function signatures.

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
"""

TRAIN_VERIFY_PROMPT = """
The actual output is:

{output_grid}

If you did not produce the correct output earlier, refine your approach and use the tool to iterate.

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

This DSL is structured natural language - you can invent functions, operators, and syntax as needed to clearly express your intent. The goal is to convey meaning in a structured, concise way rather than follow rigid syntax rules. Think of it as pseudocode that prioritizes clarity: if you need to "find all red squares that touch blue circles," just write Find("red squares", where="adjacent to blue circles"). Create helper functions like Lambda("holes", "black regions fully enclosed by shape") when concepts repeat. The DSL should read like a clear recipe for the transformation, using intuitive names and logical flow. Focus on expressing the algorithm's logic rather than worrying about exact function signatures.

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
"""

TEST_PROMPT = """
Apply your DSL program to generate the output for this input:

Input:
{input_grid}

Remember every transformation here is deterministic and reproducible. The patterns you've identified must work consistently across all examples.

Symbols may have semantic significance; properties of the symbols may convey this semantic significance. Apply the rules you've discovered based on what properties carry semantic significance.

Compositional reasoning and turn-by-turn application of rules may be important. Apply transformations in the correct sequence as you've learned from the training examples.

Apply your learned transformation rules to generate the output grid.

IMPORTANT: Provide your answer as a grid in the exact same format, with square brackets and comma-separated values:
[[0, 1, 2, ...],
 [3, 4, 5, ...],
 ...]

Make sure the dimensions are correct and include all rows.
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


class StreamingAPIClient:
    """Handles streaming API interactions with OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        self.client = OpenAI(api_key=self.api_key)
    
    def stream_response(self, messages: List[Dict]) -> Generator[StreamEvent, None, None]:
        """Stream response from the API"""
        stream = self.client.responses.create(
            model=MODEL_NAME,
            reasoning={"effort": REASONING_EFFORT},
            input=messages,
            stream=True
        )
        
        for chunk in stream:
            yield from self._process_chunk(chunk)
    
    def _process_chunk(self, chunk) -> Generator[StreamEvent, None, None]:
        """Process a streaming chunk into events"""
        if hasattr(chunk, 'output'):
            for item in chunk.output:
                if item.type == "message":
                    content = self._extract_text(item)
                    if content:
                        yield StreamEvent(type="message", content=content)
                elif item.type == "thinking":
                    thinking_content = self._extract_thinking(item)
                    if thinking_content:
                        yield StreamEvent(type="thinking", content=thinking_content)
    
    def _extract_text(self, message_item) -> str:
        """Extract text from message item"""
        if hasattr(message_item, 'content'):
            for content in message_item.content:
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'type') and content.type == "output_text":
                    return content.text if hasattr(content, 'text') else ""
        return ""
    
    def _extract_thinking(self, thinking_item) -> str:
        """Extract thinking content"""
        if hasattr(thinking_item, 'text'):
            return thinking_item.text
        elif hasattr(thinking_item, 'content'):
            return str(thinking_item.content)
        return str(thinking_item)


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
    client: StreamingAPIClient,
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
    instruction = "Analyze this first training example." if first_example else "Incorporate this training example into your DSL program."
    prompt = TRAIN_N_PROMPT.format(
        index=train_idx + 1,
        input_grid=format_grid(task_data.train[train_idx]['input']),
        output_grid=format_grid(task_data.train[train_idx]['output']),
        instruction=instruction
    )
    
    # Prepare message with labeled images
    images = [
        (f"Training {train_idx + 1} Input", input_img),
        (f"Training {train_idx + 1} Output", output_img)
    ]
    
    content = prepare_message_content(prompt, images)
    conversation_history.append({"role": "user", "content": content})
    
    # Stream response
    dsl_output = []
    for event in client.stream_response(conversation_history):
        if event.type == "thinking":
            logger.log_thinking(event.content)
        elif event.type == "message":
            dsl_output.append(event.content)
            logger.log_message(event.content)
            print(event.content, end='', flush=True)
    
    dsl = ''.join(dsl_output).strip()
    conversation_history.append({"role": "assistant", "content": dsl})
    
    logger.log_phase(f"train_{train_idx}", dsl)
    return dsl


def execute_prediction_phase(
    client: StreamingAPIClient,
    task_data: TaskData,
    idx: int,
    logger: Logger,
    conversation_history: List[Dict],
    is_test: bool = False
) -> Tuple[str, Optional[List[List[int]]]]:
    """Execute a prediction phase"""
    if is_test:
        grid = task_data.test[0]['input']
        label = "test_input"
        prompt = TEST_PROMPT.format(input_grid=format_grid(grid))
        image_label = "Test Input"
    else:
        grid = task_data.train[idx]['input']
        label = f"train_{idx}_predict"
        prompt = TEST_PROMPT.format(input_grid=format_grid(grid))
        image_label = f"Training {idx + 1} Input (Prediction)"
    
    # Create image
    input_img = create_grid_image(grid, task_data.name, label)
    
    # Prepare message
    images = [(image_label, input_img)]
    content = prepare_message_content(prompt, images)
    conversation_history.append({"role": "user", "content": content})
    
    # Stream response
    dsl_output = []
    for event in client.stream_response(conversation_history):
        if event.type == "thinking":
            logger.log_thinking(event.content)
        elif event.type == "message":
            dsl_output.append(event.content)
            logger.log_message(event.content)
            print(event.content, end='', flush=True)
    
    dsl = ''.join(dsl_output).strip()
    conversation_history.append({"role": "assistant", "content": dsl})
    
    phase_name = "test_predict" if is_test else f"train_{idx}_predict"
    logger.log_phase(phase_name, dsl)
    
    # Parse the grid output for test phase
    predicted_grid = None
    if is_test:
        predicted_grid = parse_grid_from_response(dsl)
    
    return dsl, predicted_grid


def execute_verification_phase(
    client: StreamingAPIClient,
    task_data: TaskData,
    idx: int,
    logger: Logger,
    conversation_history: List[Dict]
) -> str:
    """Execute a verification phase"""
    output_img = create_grid_image(
        task_data.train[idx]['output'],
        task_data.name,
        f"train_{idx}_verify"
    )
    
    prompt = TRAIN_VERIFY_PROMPT.format(
        output_grid=format_grid(task_data.train[idx]['output'])
    )
    
    # Prepare message
    images = [(f"Training {idx + 1} Actual Output", output_img)]
    content = prepare_message_content(prompt, images)
    conversation_history.append({"role": "user", "content": content})
    
    # Stream response
    dsl_output = []
    for event in client.stream_response(conversation_history):
        if event.type == "thinking":
            logger.log_thinking(event.content)
        elif event.type == "message":
            dsl_output.append(event.content)
            logger.log_message(event.content)
            print(event.content, end='', flush=True)
    
    dsl = ''.join(dsl_output).strip()
    conversation_history.append({"role": "assistant", "content": dsl})
    
    logger.log_phase(f"train_{idx}_verify", dsl)
    return dsl


class ARCStreamingSolver:
    """Main solver orchestrating the streaming DSL generation"""
    
    def __init__(self):
        self.client = StreamingAPIClient()
    
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
            dsl, _ = execute_prediction_phase(
                self.client, task_data, i, logger, conversation_history
            )
            num_phases += 1
            
            # Verify phase
            dsl = execute_verification_phase(
                self.client, task_data, i, logger, conversation_history
            )
            final_dsl = dsl
            num_phases += 1
        
        # Test prediction
        _, predicted_output = execute_prediction_phase(
            self.client, task_data, 0, logger, conversation_history, is_test=True
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
        solver = ARCStreamingSolver()
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