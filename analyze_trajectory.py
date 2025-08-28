#!/usr/bin/env python3
"""
Trajectory Analysis Script
Usage: python analyze_trajectory.py <trajectory_file.json>
"""
import json
import sys
from pathlib import Path

def analyze_trajectory(file_path):
    """Analyze a single trajectory file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    trajectory = data['trajectory']
    
    print("="*60)
    print(f"TRAJECTORY ANALYSIS: {Path(file_path).name}")
    print("="*60)
    
    # Basic info
    print(f"📋 Task: {trajectory['task'].split('Your task is to: ')[1].split('.')[0] if 'Your task is to: ' in trajectory['task'] else 'Unknown'}")
    print(f"📊 Total steps: {trajectory['length']}")
    print(f"🎯 Total reward: {trajectory['total_reward']}")
    print(f"✅ Success: {'Yes' if trajectory['success'] else 'No'}")
    print(f"⚙️  Max steps: {data['metadata']['config']['max_steps']}")
    print()
    
    # Action distribution
    print("📈 ACTION DISTRIBUTION")
    print("-" * 30)
    action_counts = {}
    for step in trajectory['steps']:
        action = step['agent_output']['action']
        action_type = action.split()[0] if action else 'unknown'
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{action_type:15}: {count:3d} times")
    print()
    
    # Key discoveries
    print("🔍 KEY DISCOVERIES")
    print("-" * 30)
    discoveries = []
    for step in trajectory['steps']:
        obs = step['state']['observation']
        if 'mug' in obs.lower() or 'cup' in obs.lower():
            if 'current observation is:' in obs:
                parts = obs.split('current observation is:')
                if len(parts) > 1:
                    current_obs = parts[1].split('Your admissible actions')[0].strip()[:100]
                    discoveries.append(f"Step {step['step']:2d}: {current_obs}...")
    
    if discoveries:
        for discovery in discoveries[:5]:  # Show first 5
            print(discovery)
    else:
        print("No significant discoveries found")
    print()
    
    # Success analysis
    print("🎯 SUCCESS ANALYSIS")
    print("-" * 30)
    if trajectory['success']:
        print("✅ Task marked as successful")
        # Find when success occurred
        for step in trajectory['steps']:
            if step['transition']['reward'] > 0:
                print(f"💰 First positive reward at step {step['step']}: {step['transition']['reward']}")
                break
    else:
        print("❌ Task failed")
        final_step = trajectory['steps'][-1]
        final_won = final_step['metadata']['info'].get('won', 'Unknown')
        print(f"🏁 Final 'won' status: {final_won}")
        print(f"🏁 Final reward: {final_step['transition']['reward']}")
    
    # Action sequence (first 10 and last 5)
    print("\n🎬 ACTION SEQUENCE")
    print("-" * 30)
    print("First 10 steps:")
    for step in trajectory['steps'][:10]:
        action = step['agent_output']['action']
        reward = step['transition']['reward']
        indicator = "💰" if reward > 0 else "  "
        print(f"  Step {step['step']:2d}: {action:30} {indicator}")
    
    if len(trajectory['steps']) > 15:
        print("...")
        print("Last 5 steps:")
        for step in trajectory['steps'][-5:]:
            action = step['agent_output']['action']
            reward = step['transition']['reward']
            indicator = "💰" if reward > 0 else "  "
            print(f"  Step {step['step']:2d}: {action:30} {indicator}")
    
    print()
    
    return trajectory

def compare_trajectories(file_paths):
    """Compare multiple trajectory files."""
    trajectories = []
    for path in file_paths:
        traj = analyze_trajectory(path)
        trajectories.append((Path(path).name, traj))
        print()
    
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'File':<25} {'Success':<8} {'Steps':<6} {'Reward':<8}")
    print("-" * 50)
    for name, traj in trajectories:
        success = "✅ Yes" if traj['success'] else "❌ No"
        print(f"{name:<25} {success:<8} {traj['length']:<6} {traj['total_reward']:<8}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trajectory.py <trajectory_file.json> [file2.json] ...")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    
    # Check if files exist
    for path in file_paths:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    if len(file_paths) == 1:
        analyze_trajectory(file_paths[0])
    else:
        compare_trajectories(file_paths)