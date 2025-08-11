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
import io
from contextlib import redirect_stdout
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
# from arc_solver import ARCSolver
from arc_visual_solver import ARCVisualSolver

def solve_single_task(task_file, task_number, total_tasks, use_visual=False):
    """Implementation of solve_single_task"""
    task_name = task_file.stem
    solver_type = "Visual" if use_visual else "Text"

    print(f"\n{'='*80}")
    print(f"STARTING TASK {task_number}/{total_tasks}: {task_name} [{solver_type} Solver]")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        if use_visual:
            # Use visual solver
            solver = ARCVisualSolver()
            success, prediction, num_phases = solver.solve(str(task_file))
            
            # Save prediction as image if we got one
            if prediction:
                pred_path = solver.create_grid_image(prediction, label=f"{task_name}_prediction")
                print(f"  Prediction image saved to: {pred_path}")
            
            result = {
                "task": task_name,
                "success": success,
                "time": time.time() - start_time,
                "phases": num_phases,
                "solver": "visual",
                "prediction": prediction[:3] if prediction and len(prediction) > 3 else prediction
            }
            
            if success:
                print(f"\n✅ Task {task_name} SOLVED in {result['time']:.2f}s with {num_phases} phases")
            else:
                print(f"\n❌ Task {task_name} FAILED after {result['time']:.2f}s with {num_phases} phases")
        else:
            print("Non-visual solver removed; pass -v")
            # Use original text solver
            # solver = ARCSolver()
            # success, pattern, num_prompts = solver.solve(str(task_file))
            # result = {
            #     "task": task_name,
            #     "success": success,
            #     "time": time.time() - start_time,
            #     "prompts": num_prompts,
            #     "solver": "text",
            #     "pattern": pattern[:100] if pattern else None
            # }
            
            # if success:
            #     print(f"\n✅ Task {task_name} SOLVED in {result['time']:.2f}s with {num_prompts} prompts")
            # else:
            #     print(f"\n❌ Task {task_name} FAILED after {result['time']:.2f}s with {num_prompts} prompts")
        
        return result
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n⚠️ Task {task_name} ERROR after {elapsed:.2f}s: {e}")
        return {
            "task": task_name,
            "success": False,
            "time": elapsed,
            "prompts": 0,
            "phases": 0,
            "solver": "visual" if use_visual else "text",
            "error": str(e)
        }


def run_batch_tests(num_tasks: int = 10, dataset: str = "training", parallel: int = 1, use_visual: bool = False,
                   include_tasks: list = None, exclude_tasks: list = None):
    """Run the solver on N randomly selected tasks from specified dataset"""
    
    # Path to data directory
    data_dir = Path(f"ARC-AGI-2/data/{dataset}")
    
    if not data_dir.exists():
        print(f"Error: {dataset.capitalize()} directory not found at {data_dir}")
        sys.exit(1)
    
    # Get all JSON files and randomly sample
    all_task_files = list(data_dir.glob("*.json"))
    
    # Apply include/exclude filters
    if include_tasks:
        all_task_files = [f for f in all_task_files if f.stem in include_tasks]
    elif exclude_tasks:
        all_task_files = [f for f in all_task_files if f.stem not in exclude_tasks]
    
    # Randomly sample tasks (or take all if requesting more than available)
    num_to_sample = min(num_tasks, len(all_task_files))
    task_files = random.sample(all_task_files, num_to_sample)
    
    if not task_files:
        print(f"Error: No JSON files found in {data_dir}")
        sys.exit(1)
    
    print(f"="*80)
    print(f"ARC-AGI-2 BATCH SOLVER")
    print(f"Dataset: {dataset.upper()}")
    print(f"Solver: {'Visual' if use_visual else 'Text'}")
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
            # Submit all tasks with output wrapper enabled
            futures = {
                executor.submit(solve_single_task, task_file, i, len(task_files), use_visual): task_file 
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
            result = solve_single_task(task_file, i, len(task_files), use_visual)
            results.append(result)
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
    
    # Calculate statistics
    total_time = sum(r['time'] for r in results)
    total_prompts = sum(r.get('prompts', 0) for r in results)
    total_phases = sum(r.get('phases', 0) for r in results)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(task_files)}")
    print(f"Successful: {successful} ({successful/len(task_files)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(task_files)*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    
    if use_visual:
        print(f"Total phases: {total_phases}")
    else:
        print(f"Total prompts sent: {total_prompts}")
    
    print(f"\nDetailed Results:")
    
    if use_visual:
        print(f"{'Task':<20} {'Result':<10} {'Time (s)':<10} {'Phases':<10}")
    else:
        print(f"{'Task':<20} {'Result':<10} {'Time (s)':<10} {'Prompts':<10}")
    print(f"{'-'*50}")
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        if "error" in result:
            status = "⚠️ ERROR"
        
        if use_visual:
            phases = result.get('phases', 0)
            print(f"{result['task']:<20} {status:<10} {result['time']:<10.2f} {phases:<10}")
        else:
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
    use_visual = False
    include_tasks = None
    exclude_tasks = None
    
    # Simple argument parsing
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ["-t", "--training"]:
            dataset = "training"
        elif arg in ["-e", "--evaluation"]:
            dataset = "evaluation"
        elif arg in ["-v", "--visual"]:
            use_visual = True
        elif arg in ["-p", "--parallel"]:
            if i + 1 < len(args) and args[i + 1].isdigit():
                parallel = int(args[i + 1])
                i += 1  # Skip next arg since we consumed it
            else:
                print("Error: -p/--parallel requires a number")
                sys.exit(1)
        elif arg in ["-i", "--include"]:
            if i + 1 < len(args):
                include_tasks = args[i + 1].split(',')
                i += 1  # Skip next arg since we consumed it
            else:
                print("Error: -i/--include requires a comma-separated list of task names")
                sys.exit(1)
        elif arg in ["-x", "--exclude"]:
            if i + 1 < len(args):
                exclude_tasks = args[i + 1].split(',')
                i += 1  # Skip next arg since we consumed it
            else:
                print("Error: -x/--exclude requires a comma-separated list of task names")
                sys.exit(1)
        elif arg.isdigit():
            num_tasks = int(arg)
        elif arg in ["-h", "--help"]:
            print("Usage: python run_batch.py [num_tasks] [options]")
            print("\nOptions:")
            print("  -t, --training    Use training dataset (default)")
            print("  -e, --evaluation  Use evaluation dataset")
            print("  -v, --visual      Use visual solver instead of text solver")
            print("  -p, --parallel N  Run N tasks in parallel")
            print("  -i, --include LIST  Only run these tasks (comma-separated)")
            print("  -x, --exclude LIST  Skip these tasks (comma-separated)")
            print("  -h, --help        Show this help message")
            print("\nExamples:")
            print("  python run_batch.py              # 10 random training tasks with text solver")
            print("  python run_batch.py 5 -v         # 5 random training tasks with visual solver")
            print("  python run_batch.py 5 -e         # 5 random evaluation tasks")
            print("  python run_batch.py 10 -p 3      # 10 tasks with 3 parallel workers")
            print("  python run_batch.py -i 4c416de3,89565ca0 -v  # Run specific tasks")
            print("  python run_batch.py 20 -x 4c416de3,89565ca0  # Run 20 random tasks excluding some")
            sys.exit(0)
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Use -h or --help for usage information")
            sys.exit(1)
        i += 1
    
    try:
        successful, failed = run_batch_tests(num_tasks, dataset, parallel, use_visual, include_tasks, exclude_tasks)
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()