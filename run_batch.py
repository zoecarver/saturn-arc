#!/usr/bin/env python3
"""
Batch runner for ARC-AGI-2 Solver
Runs the solver on the first 10 training tasks and reports results
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from arc_solver import ARCSolver

def run_batch_tests(num_tasks: int = 10, dataset: str = "training"):
    """Run the solver on N randomly selected tasks from specified dataset"""
    
    # Path to data directory
    data_dir = Path(f"ARC-AGI-2/data/{dataset}")
    
    if not data_dir.exists():
        print(f"Error: {dataset.capitalize()} directory not found at {data_dir}")
        sys.exit(1)
    
    # Get all JSON files and randomly sample
    all_task_files = list(data_dir.glob("*.json"))
    
    # Randomly sample tasks (or take all if requesting more than available)
    num_to_sample = min(num_tasks, len(all_task_files))
    task_files = random.sample(all_task_files, num_to_sample)
    
    if not task_files:
        print(f"Error: No JSON files found in {data_dir}")
        sys.exit(1)
    
    print(f"="*80)
    print(f"ARC-AGI-2 BATCH SOLVER")
    print(f"Dataset: {dataset.upper()}")
    print(f"Running on {len(task_files)} randomly selected tasks")
    print(f"="*80)
    
    # Track results
    results = []
    successful = 0
    failed = 0
    
    # Run each task
    for i, task_file in enumerate(task_files, 1):
        task_name = task_file.stem
        print(f"\n{'='*80}")
        print(f"TASK {i}/{len(task_files)}: {task_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Create a fresh solver for each task
            solver = ARCSolver()
            success, pattern = solver.solve(str(task_file))
            elapsed = time.time() - start_time
            
            result = {
                "task": task_name,
                "success": success,
                "time": elapsed,
                "pattern": pattern[:100] if pattern else None  # Store first 100 chars
            }
            results.append(result)
            
            if success:
                successful += 1
                print(f"\n✅ Task {task_name} SOLVED in {elapsed:.2f}s")
            else:
                failed += 1
                print(f"\n❌ Task {task_name} FAILED after {elapsed:.2f}s")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n⚠️ Task {task_name} ERROR after {elapsed:.2f}s: {e}")
            results.append({
                "task": task_name,
                "success": False,
                "time": elapsed,
                "error": str(e)
            })
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(task_files)}")
    print(f"Successful: {successful} ({successful/len(task_files)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(task_files)*100:.1f}%)")
    print(f"\nDetailed Results:")
    print(f"{'Task':<20} {'Result':<10} {'Time (s)':<10}")
    print(f"{'-'*40}")
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        if "error" in result:
            status = "⚠️ ERROR"
        print(f"{result['task']:<20} {status:<10} {result['time']:<10.2f}")
    
    # Save results to JSON
    output_file = f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return successful, failed


def main():
    """Main entry point"""
    # Parse command line arguments
    num_tasks = 10
    dataset = "training"
    
    # Simple argument parsing
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ["-t", "--training"]:
            dataset = "training"
        elif arg in ["-e", "--evaluation"]:
            dataset = "evaluation"
        elif arg.isdigit():
            num_tasks = int(arg)
        elif arg in ["-h", "--help"]:
            print("Usage: python run_batch.py [num_tasks] [options]")
            print("\nOptions:")
            print("  -t, --training    Use training dataset (default)")
            print("  -e, --evaluation  Use evaluation dataset")
            print("  -h, --help        Show this help message")
            print("\nExamples:")
            print("  python run_batch.py          # 10 random training tasks")
            print("  python run_batch.py 5        # 5 random training tasks")
            print("  python run_batch.py 5 -e     # 5 random evaluation tasks")
            sys.exit(0)
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Use -h or --help for usage information")
            sys.exit(1)
    
    try:
        successful, failed = run_batch_tests(num_tasks, dataset)
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()