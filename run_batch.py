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
from concurrent.futures import ThreadPoolExecutor, as_completed
from arc_solver import ARCSolver

def solve_single_task(task_file, task_number, total_tasks):
    """Solve a single task - used for parallel execution"""
    task_name = task_file.stem
    print(f"\n{'='*80}")
    print(f"STARTING TASK {task_number}/{total_tasks}: {task_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Create a fresh solver for each task
        solver = ARCSolver()
        success, pattern, num_prompts = solver.solve(str(task_file))
        elapsed = time.time() - start_time
        
        result = {
            "task": task_name,
            "success": success,
            "time": elapsed,
            "prompts": num_prompts,
            "pattern": pattern[:100] if pattern else None  # Store first 100 chars
        }
        
        if success:
            print(f"\n✅ Task {task_name} SOLVED in {elapsed:.2f}s with {num_prompts} prompts")
        else:
            print(f"\n❌ Task {task_name} FAILED after {elapsed:.2f}s with {num_prompts} prompts")
        
        return result
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n⚠️ Task {task_name} ERROR after {elapsed:.2f}s: {e}")
        return {
            "task": task_name,
            "success": False,
            "time": elapsed,
            "prompts": 0,
            "error": str(e)
        }


def run_batch_tests(num_tasks: int = 10, dataset: str = "training", parallel: int = 1):
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
    print(f"Parallel workers: {parallel}")
    print(f"="*80)
    
    # Track results
    results = []
    successful = 0
    failed = 0
    
    if parallel > 1:
        # Parallel execution
        print(f"\nStarting parallel execution with {parallel} workers...")
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Submit all tasks
            futures = {
                executor.submit(solve_single_task, task_file, i, len(task_files)): task_file 
                for i, task_file in enumerate(task_files, 1)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
    
    else:
        # Sequential execution (original code)
        for i, task_file in enumerate(task_files, 1):
            result = solve_single_task(task_file, i, len(task_files))
            results.append(result)
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
    
    # Calculate statistics
    total_time = sum(r['time'] for r in results)
    total_prompts = sum(r.get('prompts', 0) for r in results)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(task_files)}")
    print(f"Successful: {successful} ({successful/len(task_files)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(task_files)*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total prompts sent: {total_prompts}")
    print(f"\nDetailed Results:")
    print(f"{'Task':<20} {'Result':<10} {'Time (s)':<10} {'Prompts':<10}")
    print(f"{'-'*50}")
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        if "error" in result:
            status = "⚠️ ERROR"
        prompts = result.get('prompts', 0)
        print(f"{result['task']:<20} {status:<10} {result['time']:<10.2f} {prompts:<10}")
    
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
    parallel = 1
    
    # Simple argument parsing
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ["-t", "--training"]:
            dataset = "training"
        elif arg in ["-e", "--evaluation"]:
            dataset = "evaluation"
        elif arg in ["-p", "--parallel"]:
            if i + 1 < len(args) and args[i + 1].isdigit():
                parallel = int(args[i + 1])
                i += 1  # Skip next arg since we consumed it
            else:
                print("Error: -p/--parallel requires a number")
                sys.exit(1)
        elif arg.isdigit():
            num_tasks = int(arg)
        elif arg in ["-h", "--help"]:
            print("Usage: python run_batch.py [num_tasks] [options]")
            print("\nOptions:")
            print("  -t, --training    Use training dataset (default)")
            print("  -e, --evaluation  Use evaluation dataset")
            print("  -p, --parallel N  Run N tasks in parallel")
            print("  -h, --help        Show this help message")
            print("\nExamples:")
            print("  python run_batch.py              # 10 random training tasks")
            print("  python run_batch.py 5            # 5 random training tasks")
            print("  python run_batch.py 5 -e         # 5 random evaluation tasks")
            print("  python run_batch.py 10 -p 3      # 10 tasks with 3 parallel workers")
            sys.exit(0)
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Use -h or --help for usage information")
            sys.exit(1)
        i += 1
    
    try:
        successful, failed = run_batch_tests(num_tasks, dataset, parallel)
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()