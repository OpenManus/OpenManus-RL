#!/usr/bin/env python3
"""
Simple script to run trajectory evaluation on chat history files
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agent_trajectory_evaluator import (
    API_CONFIG,
    EvaluationPipeline
)


async def run_evaluation():
    """Run evaluation on sample trajectories"""
    
    # Configure API (you need to set these environment variables or modify here)
    API_CONFIG.update({
        "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),  # Set your API key
        "model": os.getenv("EVAL_MODEL", "gpt-4o"),
        "temperature": 0.0,
        "max_retries": 3,
        "timeout": 60
    })
    
    # Check if API key is set
    if not API_CONFIG["api_key"]:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Set input and output directories
    input_dir = "../trajectories/chat_histories"
    output_dir = "../evaluation_results"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting evaluation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {API_CONFIG['model']}")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(API_CONFIG, output_dir)
    
    # Process all trajectory files
    try:
        results = await pipeline.process_directory(
            input_dir,
            max_concurrent=3  # Limit concurrent API calls
        )
        
        # Print summary
        if results:
            avg_score = sum(r['overall_score'] for r in results) / len(results)
            success_rate = sum(1 for r in results if r['success']) / len(results)
            
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            print(f"Total trajectories evaluated: {len(results)}")
            print(f"Average score: {avg_score:.2f}/5.0")
            print(f"Success rate: {success_rate:.1%}")
            print(f"\nDetailed results saved to: {output_dir}")
            
            # Show worst performing trajectories
            sorted_results = sorted(results, key=lambda x: x['overall_score'])
            print("\nWorst performing trajectories:")
            for result in sorted_results[:3]:
                print(f"  - {result['task_id']}: Score {result['overall_score']:.2f}, Success: {result['success']}")
                if result['weaknesses']:
                    print(f"    Weaknesses: {', '.join(result['weaknesses'][:2])}")
        else:
            print("No results generated. Please check the input directory and API configuration.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(run_evaluation())