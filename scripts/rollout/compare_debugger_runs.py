#!/usr/bin/env python3
"""
Compare rollout results with and without debugger across different environments
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas as pd


def load_trajectories(jsonl_path: str) -> List[Dict]:
    """Load trajectories from JSONL file"""
    trajectories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def analyze_trajectories(trajectories: List[Dict], env_type: str = None) -> Dict:
    """Analyze trajectories and compute statistics"""
    
    # Group by environment if not specified
    env_groups = defaultdict(list)
    for traj in trajectories:
        env = traj.get('env_type', env_type or 'unknown')
        env_groups[env].append(traj)
    
    results = {}
    
    for env, env_trajs in env_groups.items():
        # Group by env_id and retry_idx
        env_attempts = defaultdict(lambda: defaultdict(list))
        
        for traj in env_trajs:
            env_id = traj.get('env_id', 0)
            retry_idx = traj.get('retry_idx', 0)
            env_attempts[env_id][retry_idx].append(traj)
        
        # Calculate statistics
        total_envs = len(env_attempts)
        success_by_retry = defaultdict(int)
        steps_by_retry = defaultdict(list)
        final_success = 0
        
        for env_id, retries in env_attempts.items():
            # Find if any retry succeeded
            env_succeeded = False
            for retry_idx, retry_trajs in sorted(retries.items()):
                # Get final state of this retry
                final_traj = max(retry_trajs, key=lambda x: x.get('step', 0))
                won = final_traj.get('won', False)
                steps = final_traj.get('step', 0) + 1
                
                if won:
                    success_by_retry[retry_idx] += 1
                    env_succeeded = True
                    steps_by_retry[retry_idx].append(steps)
                    break  # Stop at first success
                
                # Track steps even for failures
                if not env_succeeded:
                    steps_by_retry[retry_idx].append(steps)
            
            if env_succeeded:
                final_success += 1
        
        # Calculate cumulative success rates
        cumulative_success = {}
        total_success = 0
        for retry in sorted(success_by_retry.keys()):
            total_success += success_by_retry[retry]
            cumulative_success[retry] = total_success / total_envs if total_envs > 0 else 0
        
        # Calculate average steps
        avg_steps = {}
        for retry, steps_list in steps_by_retry.items():
            if steps_list:
                avg_steps[retry] = np.mean(steps_list)
        
        results[env] = {
            'total_environments': total_envs,
            'final_success_rate': final_success / total_envs if total_envs > 0 else 0,
            'success_by_retry': dict(success_by_retry),
            'cumulative_success': cumulative_success,
            'average_steps_by_retry': avg_steps,
            'max_retry_used': max(retries.keys()) if env_attempts else 0
        }
    
    return results


def compare_runs(baseline_path: str, debugger_path: str, env_type: str = None) -> None:
    """Compare baseline and debugger runs"""
    
    print("Loading trajectories...")
    baseline_trajs = load_trajectories(baseline_path)
    debugger_trajs = load_trajectories(debugger_path)
    
    print(f"Baseline: {len(baseline_trajs)} trajectory entries")
    print(f"Debugger: {len(debugger_trajs)} trajectory entries")
    print()
    
    # Analyze both sets
    baseline_stats = analyze_trajectories(baseline_trajs, env_type)
    debugger_stats = analyze_trajectories(debugger_trajs, env_type)
    
    # Compare results for each environment
    all_envs = set(baseline_stats.keys()) | set(debugger_stats.keys())
    
    for env in sorted(all_envs):
        print(f"{'='*60}")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"{'='*60}")
        
        base = baseline_stats.get(env, {})
        debug = debugger_stats.get(env, {})
        
        if base:
            print("\nBASELINE (without debugger):")
            print(f"  Total Environments: {base['total_environments']}")
            print(f"  Success Rate: {base['final_success_rate']:.2%}")
            if base['average_steps_by_retry']:
                avg_steps = list(base['average_steps_by_retry'].values())[0]
                print(f"  Average Steps: {avg_steps:.1f}")
        
        if debug:
            print("\nDEBUGGER (with retry logic):")
            print(f"  Total Environments: {debug['total_environments']}")
            print(f"  Final Success Rate: {debug['final_success_rate']:.2%}")
            print(f"  Max Retries Used: {debug['max_retry_used']}")
            
            print("\n  Success by Retry:")
            for retry in sorted(debug['cumulative_success'].keys()):
                cum_rate = debug['cumulative_success'][retry]
                new_success = debug['success_by_retry'].get(retry, 0)
                print(f"    Retry {retry}: {cum_rate:.2%} cumulative (+{new_success} new)")
            
            if debug['average_steps_by_retry']:
                print("\n  Average Steps by Retry:")
                for retry, steps in sorted(debug['average_steps_by_retry'].items()):
                    print(f"    Retry {retry}: {steps:.1f} steps")
        
        # Calculate improvement
        if base and debug:
            base_rate = base['final_success_rate']
            debug_rate = debug['final_success_rate']
            
            if base_rate > 0:
                improvement = (debug_rate - base_rate) / base_rate * 100
                absolute_improvement = (debug_rate - base_rate) * 100
                
                print(f"\nIMPROVEMENT:")
                print(f"  Absolute: +{absolute_improvement:.1f}%")
                print(f"  Relative: +{improvement:.1f}%")
            else:
                print(f"\nIMPROVEMENT:")
                print(f"  From 0% to {debug_rate:.2%}")
        
        print()
    
    # Overall summary
    print(f"{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    # Calculate weighted averages
    total_base_envs = sum(s['total_environments'] for s in baseline_stats.values())
    total_debug_envs = sum(s['total_environments'] for s in debugger_stats.values())
    
    if total_base_envs > 0:
        weighted_base_success = sum(
            s['final_success_rate'] * s['total_environments'] 
            for s in baseline_stats.values()
        ) / total_base_envs
        print(f"Baseline Average Success: {weighted_base_success:.2%}")
    
    if total_debug_envs > 0:
        weighted_debug_success = sum(
            s['final_success_rate'] * s['total_environments'] 
            for s in debugger_stats.values()
        ) / total_debug_envs
        print(f"Debugger Average Success: {weighted_debug_success:.2%}")
        
        if total_base_envs > 0:
            overall_improvement = (weighted_debug_success - weighted_base_success) * 100
            print(f"Overall Improvement: +{overall_improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare debugger effectiveness")
    parser.add_argument("baseline", help="Path to baseline trajectory JSONL")
    parser.add_argument("debugger", help="Path to debugger trajectory JSONL")
    parser.add_argument("--env", choices=["alfworld", "gaia", "webshop"],
                       help="Filter by environment type")
    parser.add_argument("--output", help="Save comparison to file")
    
    args = parser.parse_args()
    
    # Redirect output if specified
    import sys
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    try:
        compare_runs(args.baseline, args.debugger, args.env)
    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
