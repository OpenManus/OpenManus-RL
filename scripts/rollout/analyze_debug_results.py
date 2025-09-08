#!/usr/bin/env python3
"""
Analyze debug results from alfworld_debugger runs
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import numpy as np


def load_trajectories(jsonl_path: str) -> List[Dict]:
    """Load trajectories from JSONL file"""
    trajectories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def load_debug_analyses(debug_dir: str) -> List[Dict]:
    """Load all debug analysis JSON files from directory"""
    analyses = []
    if not os.path.exists(debug_dir):
        return analyses
    
    for filename in os.listdir(debug_dir):
        if filename.endswith('.json'):
            with open(os.path.join(debug_dir, filename), 'r') as f:
                data = json.load(f)
                analyses.append(data)
    return analyses


def analyze_retry_effectiveness(trajectories: List[Dict]) -> Dict:
    """Analyze the effectiveness of retry attempts"""
    
    # Group by environment
    env_attempts = defaultdict(list)
    for traj in trajectories:
        env_id = traj.get('env_id', 0)
        retry_idx = traj.get('retry_idx', 0)
        won = traj.get('won', False)
        step = traj.get('step', 0)
        
        env_attempts[env_id].append({
            'retry': retry_idx,
            'step': step,
            'won': won
        })
    
    # Analyze success by retry attempt
    retry_success = defaultdict(lambda: {'total': 0, 'won': 0})
    for env_id, attempts in env_attempts.items():
        # Get final outcome for each retry
        retry_outcomes = defaultdict(lambda: {'won': False, 'max_step': 0})
        for attempt in attempts:
            retry = attempt['retry']
            retry_outcomes[retry]['won'] = retry_outcomes[retry]['won'] or attempt['won']
            retry_outcomes[retry]['max_step'] = max(retry_outcomes[retry]['max_step'], attempt['step'])
        
        for retry, outcome in retry_outcomes.items():
            retry_success[retry]['total'] += 1
            if outcome['won']:
                retry_success[retry]['won'] += 1
    
    # Calculate success rates
    success_by_retry = {}
    cumulative_success = 0
    total_envs = retry_success[0]['total'] if 0 in retry_success else 0
    
    for retry in sorted(retry_success.keys()):
        if retry_success[retry]['total'] > 0:
            success_rate = retry_success[retry]['won'] / retry_success[retry]['total']
            cumulative_success += retry_success[retry]['won']
            cumulative_rate = cumulative_success / total_envs if total_envs > 0 else 0
            
            success_by_retry[retry] = {
                'success_rate': success_rate,
                'cumulative_success_rate': cumulative_rate,
                'new_successes': retry_success[retry]['won'],
                'total_attempts': retry_success[retry]['total']
            }
    
    return {
        'success_by_retry': success_by_retry,
        'total_environments': total_envs,
        'final_success_rate': cumulative_success / total_envs if total_envs > 0 else 0
    }


def analyze_failure_types(debug_analyses: List[Dict]) -> Dict:
    """Analyze the distribution of failure types"""
    
    failure_types = defaultdict(int)
    failure_steps = []
    critical_steps = []
    
    for analysis_file in debug_analyses:
        analysis = analysis_file.get('analysis', {})
        failure_type = analysis.get('failure_type', 'unknown')
        failure_step = analysis.get('failure_step', -1)
        critical_step = analysis.get('critical_step', -1)
        
        failure_types[failure_type] += 1
        if failure_step >= 0:
            failure_steps.append(failure_step)
        if critical_step >= 0:
            critical_steps.append(critical_step)
    
    return {
        'failure_type_distribution': dict(failure_types),
        'avg_failure_step': np.mean(failure_steps) if failure_steps else 0,
        'avg_critical_step': np.mean(critical_steps) if critical_steps else 0,
        'failure_step_range': (min(failure_steps), max(failure_steps)) if failure_steps else (0, 0)
    }


def generate_report(trajectories: List[Dict], debug_analyses: List[Dict]) -> str:
    """Generate a comprehensive analysis report"""
    
    retry_analysis = analyze_retry_effectiveness(trajectories)
    failure_analysis = analyze_failure_types(debug_analyses)
    
    report = []
    report.append("=" * 60)
    report.append("ALFWORLD DEBUGGER ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Retry effectiveness
    report.append("RETRY EFFECTIVENESS")
    report.append("-" * 40)
    report.append(f"Total Environments: {retry_analysis['total_environments']}")
    report.append(f"Final Success Rate: {retry_analysis['final_success_rate']:.2%}")
    report.append("")
    
    report.append("Success Rate by Retry Attempt:")
    for retry, stats in sorted(retry_analysis['success_by_retry'].items()):
        report.append(f"  Retry {retry}:")
        report.append(f"    - Success Rate: {stats['success_rate']:.2%}")
        report.append(f"    - Cumulative Success: {stats['cumulative_success_rate']:.2%}")
        report.append(f"    - New Successes: {stats['new_successes']}/{stats['total_attempts']}")
    report.append("")
    
    # Failure type analysis
    if debug_analyses:
        report.append("FAILURE TYPE ANALYSIS")
        report.append("-" * 40)
        report.append("Failure Type Distribution:")
        total_failures = sum(failure_analysis['failure_type_distribution'].values())
        for failure_type, count in sorted(failure_analysis['failure_type_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
            percentage = (count / total_failures * 100) if total_failures > 0 else 0
            report.append(f"  - {failure_type}: {count} ({percentage:.1f}%)")
        report.append("")
        
        report.append(f"Average Failure Step: {failure_analysis['avg_failure_step']:.1f}")
        report.append(f"Average Critical Step: {failure_analysis['avg_critical_step']:.1f}")
        report.append(f"Failure Step Range: {failure_analysis['failure_step_range'][0]}-{failure_analysis['failure_step_range'][1]}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze AlfWorld debugger results")
    parser.add_argument("--trajectory_file", required=True, help="Path to trajectory JSONL file")
    parser.add_argument("--debug_dir", help="Path to debug analysis directory")
    parser.add_argument("--output", help="Output file for report (default: print to console)")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading trajectories from {args.trajectory_file}...")
    trajectories = load_trajectories(args.trajectory_file)
    print(f"Loaded {len(trajectories)} trajectory entries")
    
    debug_analyses = []
    if args.debug_dir:
        print(f"Loading debug analyses from {args.debug_dir}...")
        debug_analyses = load_debug_analyses(args.debug_dir)
        print(f"Loaded {len(debug_analyses)} debug analysis files")
    
    # Generate report
    report = generate_report(trajectories, debug_analyses)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
