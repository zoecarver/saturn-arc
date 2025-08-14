#!/usr/bin/env python3
"""
Claude Tool-Heavy Numerical Solver for ARC-AGI
Uses structured tool orchestration without visual components
Focus on simplicity and clean logic
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import anthropic

# Global configuration for API calls
API_CONFIG = {
    "model": "claude-opus-4-1-20250805",
    "max_tokens": 32000,
    "temperature": 0.2,  # Lower temperature for more consistent outputs
    "thinking": {
        "type": "enabled",
        "budget_tokens": 16000  # Allocate tokens for thinking (min 1024)
    }
}


class ClaudeToolSolver:
    """Solver using Claude API with heavy tool usage and no visual components"""
    
    def __init__(self):
        """Initialize the Claude solver with API credentials"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.current_task_name = None
        self.submitted_answer = None
        self.max_tool_calls = 20
        self.max_iterations = 2
        self.messages = []  # Track messages across phases
        
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from a JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def diff_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Compare two grids and return differences"""
        differences = []
        rows1, cols1 = len(grid1), len(grid1[0]) if grid1 else 0
        rows2, cols2 = len(grid2), len(grid2[0]) if grid2 else 0
        
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
        
        total_cells = max(rows1 * cols1, rows2 * cols2)
        return {
            "grid1_size": f"{rows1}x{cols1}",
            "grid2_size": f"{rows2}x{cols2}",
            "different_cells": len(differences),
            "match_percentage": round(100 * (1 - len(differences) / total_cells), 1) if total_cells > 0 else 0,
            "differences": differences[:10]  # Limit to first 10 differences for brevity
        }
    
    def create_tools(self) -> List[Dict[str, Any]]:
        """Create tool definitions for Claude API"""
        return [
            {
                "name": "get_train_input",
                "description": "Get a specific training example's input grid as a 2D array",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "example_num": {
                            "type": "integer",
                            "description": "Training example number (0-based index)"
                        }
                    },
                    "required": ["example_num"]
                }
            },
            {
                "name": "get_train_output",
                "description": "Get a specific training example's output grid as a 2D array",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "example_num": {
                            "type": "integer",
                            "description": "Training example number (0-based index)"
                        }
                    },
                    "required": ["example_num"]
                }
            },
            {
                "name": "get_test_input",
                "description": "Get the test input grid as a 2D array",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "diff_grids",
                "description": "Compare two grids numerically and return differences",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "grid1": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "description": "First grid to compare"
                        },
                        "grid2": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "description": "Second grid to compare"
                        }
                    },
                    "required": ["grid1", "grid2"]
                }
            },
            {
                "name": "submit_test_output",
                "description": "Submit your final answer for the test output grid",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "grid": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "description": "2D array representing your predicted test output"
                        }
                    },
                    "required": ["grid"]
                }
            }
        ]
    
    def handle_tool_call(self, tool_name: str, tool_input: Dict[str, Any], task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call and return the result"""
        
        if tool_name == "get_train_input":
            example_num = tool_input["example_num"]
            if 0 <= example_num < len(task_data["train"]):
                return {
                    "grid": task_data["train"][example_num]["input"],
                    "status": "success"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Invalid example number {example_num}. Valid range is 0-{len(task_data['train'])-1}"
                }
        
        elif tool_name == "get_train_output":
            example_num = tool_input["example_num"]
            if 0 <= example_num < len(task_data["train"]):
                return {
                    "grid": task_data["train"][example_num]["output"],
                    "status": "success"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Invalid example number {example_num}. Valid range is 0-{len(task_data['train'])-1}"
                }
        
        elif tool_name == "get_test_input":
            return {
                "grid": task_data["test"][0]["input"],
                "status": "success"
            }
        
        elif tool_name == "diff_grids":
            result = self.diff_grids(tool_input["grid1"], tool_input["grid2"])
            result["status"] = "success"
            return result
        
        elif tool_name == "submit_test_output":
            self.submitted_answer = tool_input["grid"]
            
            # Check against actual test output
            actual_output = task_data["test"][0]["output"]
            if self.submitted_answer == actual_output:
                return {
                    "status": "success",
                    "message": "SUCCESS! Your answer is correct!",
                    "correct": True
                }
            else:
                return {
                    "status": "success",
                    "message": "Your answer does not match the expected output. Try again with a different approach.",
                    "correct": False
                }
        
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}
    
    def extract_confidence(self, response_text: str) -> str:
        """Extract confidence level from response"""
        text_upper = response_text.upper()
        if "CONFIDENCE: HIGH" in text_upper:
            return "HIGH"
        elif "CONFIDENCE: MEDIUM" in text_upper:
            return "MEDIUM"
        elif "CONFIDENCE: LOW" in text_upper:
            return "LOW"
        return "UNKNOWN"
    
    def process_and_log_response(self, response, round_num: int = 0) -> None:
        """Process response content and log thinking and text"""
        print(f"\n--- Round {round_num} Response ---")
        
        # Process each content block
        for content in response.content:
            if content.type == "thinking":
                # Log thinking content
                print(f"🧠 THINKING:")
                print("-" * 40)
                # Truncate if too long, but show enough to be useful
                thinking_text = content.text if hasattr(content, 'text') else str(content)
                if len(thinking_text) > 5000:
                    print(thinking_text[:5000] + "...[truncated]")
                else:
                    print(thinking_text)
                print("-" * 40)
            
            elif content.type == "text":
                # Log regular text response
                print(f"💬 RESPONSE TEXT:")
                print("-" * 40)
                text = content.text if hasattr(content, 'text') else str(content)
                if len(text) > 5000:
                    print(text[:5000] + "...[truncated]")
                else:
                    print(text)
                print("-" * 40)
            
            elif content.type == "tool_use":
                # Tool usage will be logged separately
                pass
    
    def solve_with_structured_approach(self, task_data: Dict[str, Any]) -> Tuple[bool, Optional[List[List[int]]]]:
        """Solve using structured sequential tool usage"""
        
        num_examples = len(task_data["train"])
        
        # Create the structured prompt
        prompt = f"""You must solve this grid transformation puzzle using EXACTLY this sequence:

PHASE 1: Initial Pattern Discovery
1. Call get_train_input(0) - examine first training input
2. Call get_train_output(0) - examine first training output  
3. Identify what changed between input and output
4. Form initial hypothesis about the transformation rule

PHASE 2: Hypothesis Testing (for examples 1 to {num_examples-1})
5. For each remaining example:
   - Call get_train_input(i)
   - Apply your hypothesis to predict the output
   - Call get_train_output(i) to check actual output
   - If prediction differs, use diff_grids to understand why
   - Refine your hypothesis based on differences

PHASE 3: Rule Formulation
6. Ensure your rule works for ALL {num_examples} examples
7. Convert observations into precise algorithmic steps:
   - For each cell at position (row, col):
   - If [specific condition], then [specific action]
   - Be exact with numbers and positions

PHASE 4: Test Application
8. Call get_test_input() 
9. Apply your final rule step-by-step
10. Rate your confidence:
    CONFIDENCE: HIGH (rule works perfectly on all examples)
    CONFIDENCE: MEDIUM (rule works but some uncertainty)
    CONFIDENCE: LOW (rule has gaps)

PHASE 5: Submission
11. Use submit_test_output with your answer

IMPORTANT: You MUST submit an answer using the submit_test_output tool.
The answer must be in the correct format - a 2D array like:
[[0,1,2],[3,4,5],[6,7,8]]

Each row should be an array of integers (0-9).
DO NOT just describe the answer - you MUST call submit_test_output with the grid.

Do NOT skip steps. Be methodical and precise.
Number of training examples: {num_examples}"""

        # Initialize or append to messages
        if not self.messages:
            self.messages = [{"role": "user", "content": prompt}]
        else:
            self.messages.append({"role": "user", "content": prompt})
        
        # Make API call with tools
        response = self.client.messages.create(
            **API_CONFIG,
            messages=self.messages,
            tools=self.create_tools(),
            tool_choice={"type": "auto"}
        )
        
        # Log initial response
        self.process_and_log_response(response, round_num=0)
        
        # Process tool calls iteratively
        tool_call_count = 0
        max_rounds = 50
        
        for round_num in range(max_rounds):
            # Check if response has tool calls
            if not response.stop_reason == "tool_use":
                break
            
            # Process tool calls
            tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_call_count += 1
                    print(f"🔧 Tool call {tool_call_count}: {content.name}")
                    
                    result = self.handle_tool_call(content.name, content.input, task_data)
                    tool_results.append({
                        "tool_use_id": content.id,
                        "content": json.dumps(result)
                    })
                    
                    # Check if submission was made
                    if content.name == "submit_test_output" and result.get("correct"):
                        print("✅ Correct answer submitted!")
                        return True, self.submitted_answer
            
            if not tool_results:
                break
            
            # Add assistant message and tool results to conversation
            self.messages.append({
                "role": "assistant",
                "content": response.content
            })
            self.messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]} 
                          for tr in tool_results]
            })
            
            # Check if we've exceeded tool call budget
            if tool_call_count >= self.max_tool_calls:
                self.messages.append({
                    "role": "user",
                    "content": "You've reached the tool call limit. Please submit your final answer now."
                })
            
            # Continue conversation
            response = self.client.messages.create(
                **API_CONFIG,
                messages=self.messages,
                tools=self.create_tools(),
                tool_choice={"type": "auto"}
            )
            
            # Log the new response
            self.process_and_log_response(response, round_num=round_num + 1)
        
        return False, self.submitted_answer
    
    def solve_with_confidence_gates(self, task_data: Dict[str, Any]) -> Tuple[bool, Optional[List[List[int]]]]:
        """Solve with confidence validation gates"""
        
        # First attempt with structured approach
        success, answer = self.solve_with_structured_approach(task_data)
        
        if success:
            return True, answer
        
        # If first attempt failed, try with validation emphasis
        validation_prompt = f"""Your previous attempt was incorrect. Let's be more careful.

Before submitting, validate your answer:
1. DIMENSIONS: Does output size match typical outputs from training?
2. COMPLETENESS: Did you apply the transformation to ALL relevant parts?
3. CONSISTENCY: Does this follow the EXACT same rule as training examples?

Re-examine the training examples more carefully:
- Use get_train_input and get_train_output for ALL {len(task_data['train'])} examples
- Use diff_grids to verify your understanding
- Only submit when CONFIDENCE: HIGH

What patterns did you miss in your first attempt?

IMPORTANT: You MUST submit a new answer using the submit_test_output tool (after iterating with other tools and refining your approach).
The answer must be a 2D array like: [[0,1,2],[3,4,5],[6,7,8]]
DO NOT just describe what the answer should be - actually call submit_test_output."""

        # Continue with existing messages to maintain context
        self.messages.append({"role": "user", "content": validation_prompt})
        
        response = self.client.messages.create(
            **API_CONFIG,
            messages=self.messages,
            tools=self.create_tools(),
            tool_choice={"type": "auto"}
        )
        
        # Log initial response for second attempt
        print("\n=== SECOND ATTEMPT (After Validation Prompt) ===")
        self.process_and_log_response(response, round_num=0)
        
        # Process second attempt
        tool_call_count = 0
        for round_num in range(25):
            if not response.stop_reason == "tool_use":
                break
            
            tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_call_count += 1
                    result = self.handle_tool_call(content.name, content.input, task_data)
                    tool_results.append({
                        "tool_use_id": content.id,
                        "content": json.dumps(result)
                    })
                    
                    if content.name == "submit_test_output" and result.get("correct"):
                        return True, self.submitted_answer
            
            if not tool_results:
                break
            
            self.messages.append({"role": "assistant", "content": response.content})
            self.messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]} 
                          for tr in tool_results]
            })
            
            response = self.client.messages.create(
                **API_CONFIG,
                messages=self.messages,
                tools=self.create_tools(),
                tool_choice={"type": "auto"}
            )
            
            # Log the new response
            self.process_and_log_response(response, round_num=round_num + 1)
        
        return False, self.submitted_answer
    
    def solve(self, task_file: str) -> Tuple[bool, Optional[List[List[int]]]]:
        """Main solving method"""
        # Extract task name
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        
        # Load task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Training examples: {len(task['train'])}, Test examples: {len(task['test'])}")
        
        # Reset state
        self.submitted_answer = None
        self.messages = []  # Reset messages for new task
        
        # Try solving with confidence gates
        success, prediction = self.solve_with_confidence_gates(task)
        
        # Report results
        print(f"\n{'='*80}")
        print(f"Task: {self.current_task_name}")
        print(f"Result: {'SUCCESS ✅' if success else 'FAILED ❌'}")
        
        if prediction:
            print(f"Prediction shape: {len(prediction)}x{len(prediction[0]) if prediction else 0}")
        
        print(f"{'='*80}")
        
        return success, prediction


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python claude_tool_solver.py <task_json_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        solver = ClaudeToolSolver()
        success, prediction = solver.solve(task_file)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()