#!/usr/bin/env python3
"""
DSL Executor
Applies DSL programs to input grids using GPT-5
"""

import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Model configuration
MODEL_NAME = "gpt-5"
REASONING_EFFORT = "low"  # Low reasoning for simple execution

# DSL Execution prompt
EXECUTE_DSL_PROMPT = """You are a DSL interpreter. Execute the given DSL program on the input grid and return ONLY the output grid.

DSL Program:
{dsl_program}

Input Grid:
{input_grid}

Instructions:
1. Apply the DSL program to transform the input grid
2. Output ONLY the resulting grid as an array of arrays
3. Use square brackets and comma-separated values
4. Each row should be on its own line

Example output format:
[[0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]]

IMPORTANT: Output ONLY the grid, no explanations or other text.
"""


class DSLExecutor:
    """Executes DSL programs on grids using GPT-5"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the DSL executor"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        self.client = OpenAI(api_key=self.api_key)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def parse_grid(self, response: str) -> Optional[List[List[int]]]:
        """Parse a grid from the response"""
        import re
        
        # Find all bracket-enclosed sequences
        pattern = r'\[[\d,\s-]+\]'
        matches = re.findall(pattern, response)
        
        grid = []
        for match in matches:
            # Remove brackets and parse numbers
            numbers = match.strip('[]').replace(' ', '').split(',')
            row = [int(n) for n in numbers if n or n == '0']
            if row:
                grid.append(row)
        
        return grid if grid else None
    
    def execute(self, input_grid: List[List[int]], dsl_program: str) -> Optional[List[List[int]]]:
        """
        Execute a DSL program on an input grid
        
        Args:
            input_grid: The input grid to transform
            dsl_program: The DSL program to execute
            
        Returns:
            The transformed grid, or None if execution failed
        """
        # Prepare the prompt
        prompt = EXECUTE_DSL_PROMPT.format(
            dsl_program=dsl_program,
            input_grid=self.format_grid(input_grid)
        )
        
        # Create message
        messages = [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ]
        
        try:
            # Call GPT-5 with low reasoning
            response = self.client.responses.create(
                model=MODEL_NAME,
                reasoning={"effort": REASONING_EFFORT},
                input=messages
            )
            
            # Extract output text
            output_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                output_text = response.output_text
            else:
                # Try to extract from output items
                for item in response.output:
                    if item.type == "message" and hasattr(item, 'content'):
                        for content_item in item.content:
                            if content_item.type == "output_text" and hasattr(content_item, 'text'):
                                output_text = content_item.text
                                break
            
            # Parse the grid from the response
            if output_text:
                return self.parse_grid(output_text)
            else:
                print("Warning: No output text received from API")
                return None
                
        except Exception as e:
            print(f"Error executing DSL: {e}")
            return None


def create_dsl_executor_tool():
    """Create the tool definition for DSL execution"""
    return {
        "type": "function",
        "name": "execute_dsl",
        "description": "Execute a DSL program on an input grid to produce an output grid",
        "parameters": {
            "type": "object",
            "properties": {
                "input_grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "The input grid to transform"
                },
                "dsl_program": {
                    "type": "string",
                    "description": "The DSL program to execute on the input"
                }
            },
            "required": ["input_grid", "dsl_program"]
        }
    }


# Standalone usage example
def main():
    """Example usage of the DSL executor"""
    import sys
    
    # Example DSL and grid
    example_dsl = """
    program {
        grid = Input()
        // Flip all non-zero values to 9
        for cell in grid {
            if cell != 0 {
                cell = 9
            }
        }
        Output(grid)
    }
    """
    
    example_grid = [
        [0, 1, 2],
        [3, 0, 4],
        [5, 6, 0]
    ]
    
    print("DSL Executor Example")
    print("=" * 40)
    print("Input Grid:")
    for row in example_grid:
        print(row)
    print("\nDSL Program:")
    print(example_dsl)
    print("\nExecuting...")
    
    executor = DSLExecutor()
    result = executor.execute(example_grid, example_dsl)
    
    if result:
        print("\nOutput Grid:")
        for row in result:
            print(row)
    else:
        print("\nExecution failed")


if __name__ == "__main__":
    main()