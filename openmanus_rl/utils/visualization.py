"""
Visualization utilities for OpenManus agent trajectories.
Replaces dependency on ragen.utils.plot with custom implementation.
"""

import os
import re
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image


def parse_llm_output(text: str, strategy: str = "raw") -> Dict[str, Any]:
    """
    Parse LLM output text according to different strategies.
    
    Args:
        text: Raw text output from LLM
        strategy: Parsing strategy, options:
            - "raw": Return raw text
            - "action_response": Extract <action> and <response> tags
            - "react": Parse ReAct format (Thought, Action, Observation)
    
    Returns:
        Dictionary with parsed components
    """
    result = {"raw": text}
    
    if strategy == "raw":
        return result
    
    if strategy == "action_response":
        # Extract action and response tags
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        
        result["action"] = action_match.group(1).strip() if action_match else None
        result["response"] = response_match.group(1).strip() if response_match else None
        return result
    
    if strategy == "react":
        # Parse ReAct format (Thought, Action, Observation)
        thought_match = re.search(r'(?:Think:|Thought:)\s*(.*?)(?:Act:|Action:|$)', text, re.DOTALL)
        action_match = re.search(r'(?:Act:|Action:)\s*(.*?)(?:Obs:|Observation:|$)', text, re.DOTALL)
        obs_match = re.search(r'(?:Obs:|Observation:)\s*(.*?)$', text, re.DOTALL)
        
        result["thought"] = thought_match.group(1).strip() if thought_match else None
        result["action"] = action_match.group(1).strip() if action_match else None
        result["observation"] = obs_match.group(1).strip() if obs_match else None
        return result
    
    # Default case
    return result


def save_trajectory_to_output(
    trajectories: List[Dict[str, Any]],
    save_dir: str,
    format: str = "html",
    prefix: str = "trajectory"
) -> List[str]:
    """
    Save agent trajectories to output files.
    
    Args:
        trajectories: List of trajectory dictionaries
        save_dir: Directory to save output files
        format: Output format (html, json, txt)
        prefix: Filename prefix
    
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for i, traj in enumerate(trajectories):
        filename = f"{prefix}_{timestamp}_{i}"
        
        if format == "json":
            # Save as JSON
            filepath = os.path.join(save_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(traj, f, indent=2)
            saved_files.append(filepath)
            
        elif format == "html":
            # Save as HTML with visualization
            filepath = os.path.join(save_dir, f"{filename}.html")
            _create_html_visualization(traj, filepath)
            saved_files.append(filepath)
            
        elif format == "txt":
            # Save as plain text
            filepath = os.path.join(save_dir, f"{filename}.txt")
            with open(filepath, 'w') as f:
                f.write(_trajectory_to_text(traj))
            saved_files.append(filepath)
            
        # If trajectory contains state images, save them too
        if "state" in traj and isinstance(traj["state"], list):
            state_dir = os.path.join(save_dir, f"{filename}_states")
            os.makedirs(state_dir, exist_ok=True)
            
            for j, state_img in enumerate(traj["state"]):
                if isinstance(state_img, np.ndarray):
                    img_path = os.path.join(state_dir, f"state_{j:03d}.png")
                    Image.fromarray(state_img).save(img_path)
                    saved_files.append(img_path)
    
    return saved_files


def _trajectory_to_text(trajectory: Dict[str, Any]) -> str:
    """Convert a trajectory to formatted text."""
    text = "AGENT TRAJECTORY\n" + "="*50 + "\n\n"
    
    # Add answers/responses
    if "answer" in trajectory and isinstance(trajectory["answer"], list):
        for i, answer in enumerate(trajectory["answer"]):
            text += f"STEP {i+1}:\n"
            text += f"Agent Response:\n{answer}\n\n"
    
    # Add parsed responses if available
    if "parsed_response" in trajectory and isinstance(trajectory["parsed_response"], list):
        text += "\nPARSED RESPONSES\n" + "-"*50 + "\n\n"
        for i, parsed in enumerate(trajectory["parsed_response"]):
            text += f"Step {i+1} Parsed:\n"
            for key, value in parsed.items():
                if value:
                    text += f"  {key}: {value}\n"
            text += "\n"
    
    return text


def _create_html_visualization(trajectory: Dict[str, Any], filepath: str) -> None:
    """Create HTML visualization of a trajectory."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Trajectory Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .step { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .response { white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 3px; }
            .parsed { margin-top: 10px; color: #555; }
            .thought { color: #0066cc; }
            .action { color: #cc6600; }
            .observation { color: #006600; }
            img { max-width: 100%; max-height: 300px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Agent Trajectory Visualization</h1>
    """
    
    # Process each step
    steps = len(trajectory.get("answer", []))
    for i in range(steps):
        html += f'<div class="step"><h3>Step {i+1}</h3>'
        
        # Add agent response
        if "answer" in trajectory and i < len(trajectory["answer"]):
            html += f'<div class="response">{trajectory["answer"][i]}</div>'
        
        # Add parsed components
        if "parsed_response" in trajectory and i < len(trajectory["parsed_response"]):
            parsed = trajectory["parsed_response"][i]
            html += '<div class="parsed">'
            
            if "thought" in parsed and parsed["thought"]:
                html += f'<div class="thought"><strong>Thought:</strong> {parsed["thought"]}</div>'
            
            if "action" in parsed and parsed["action"]:
                html += f'<div class="action"><strong>Action:</strong> {parsed["action"]}</div>'
                
            if "observation" in parsed and parsed["observation"]:
                html += f'<div class="observation"><strong>Observation:</strong> {parsed["observation"]}</div>'
                
            html += '</div>'
        
        # Add state image if available
        if "state" in trajectory and isinstance(trajectory["state"], list) and i < len(trajectory["state"]):
            html += f'<div><img src="{os.path.basename(filepath)}_states/state_{i:03d}.png" alt="State {i}"></div>'
            
        html += '</div>'
    
    html += """
    </body>
    </html>
    """
    
    with open(filepath, 'w') as f:
        f.write(html)


# Compatibility alias for easier migration
def plot_trajectory(*args, **kwargs):
    """Compatibility function for legacy code."""
    return save_trajectory_to_output(*args, **kwargs) 