#!/usr/bin/env python3
"""
Simplified API Interface for Agent Error Detection System V2
For integration with trajectory correction/reloading systems
Using latest error definitions and analysis modules

Author: Zijia Liu
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import os

# Ensure error definitions loader can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the latest detection modules
from analysis_v5_error_detection import ErrorTypeDetector
from analysis_phase2_v3_critical import CriticalErrorAnalyzer
from error_definitions_loader_v3 import ErrorDefinitionsLoader


class AgentErrorDetectorAPI:
    """
    Main API class for error detection in agent trajectories

    This class provides a simple interface for external systems to:
    1. Detect all errors in a failed trajectory (Phase 1)
    2. Identify the critical error that caused failure (Phase 2)
    3. Get correction guidance for trajectory reload
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-2025-04-14"):
        """
        Initialize the error detector

        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
        """
        self.api_config = {
            "base_url": "https://api.openai.com/v1/chat/completions",
            "api_key": api_key,
            "model": model,
            "temperature": 0.0,
            "max_retries": 3,
            "timeout": 60
        }

        self.phase1_detector = ErrorTypeDetector(self.api_config)
        self.phase2_analyzer = CriticalErrorAnalyzer(self.api_config)
        self.error_loader = ErrorDefinitionsLoader()

    async def analyze_trajectory(self, trajectory_json: Dict) -> Dict:
        """
        Complete analysis pipeline: Phase 1 + Phase 2

        Args:
            trajectory_json: Complete trajectory data

        Returns:
            Combined results with step-by-step errors and critical error
        """
        # Phase 1: Detect all errors
        phase1_results = await self.detect_errors(trajectory_json)

        # Phase 2: Find critical error
        phase2_results = await self.find_critical_error(phase1_results, trajectory_json)

        return {
            'phase1_errors': phase1_results,
            'critical_error': phase2_results.get('critical_error'),
            'task_success': phase1_results.get('task_success', False),
            'environment': phase1_results.get('environment', 'unknown')
        }

    async def detect_errors(self, trajectory_json: Dict) -> Dict:
        """
        Phase 1: Detect all errors in trajectory

        Args:
            trajectory_json: Complete trajectory data

        Returns:
            Step-by-step error analysis
        """
        # Parse trajectory
        trajectory_data = self.phase1_detector.parse_trajectory_from_dict(trajectory_json)

        # Analyze for errors
        results = await self.phase1_detector.analyze_trajectory(trajectory_data)

        return results

    async def find_critical_error(self, phase1_results: Dict, trajectory_json: Dict) -> Dict:
        """
        Phase 2: Identify critical failure point

        Args:
            phase1_results: Results from Phase 1 analysis
            trajectory_json: Original trajectory data

        Returns:
            Critical error identification with correction guidance
        """
        # Find critical error
        critical_error = await self.phase2_analyzer.identify_critical_error(
            phase1_results,
            trajectory_json
        )

        return critical_error

    def get_error_definitions(self, module: Optional[str] = None) -> Dict:
        """
        Get error type definitions

        Args:
            module: Specific module name or None for all

        Returns:
            Error definitions with examples
        """
        if module:
            return self.error_loader.get_module_definitions(module)
        else:
            all_defs = {}
            for mod in ['memory', 'reflection', 'planning', 'action', 'system', 'others']:
                all_defs[mod] = self.error_loader.get_module_definitions(mod)
            return all_defs


# Synchronous wrapper functions for easier integration
def analyze_trajectory_sync(trajectory_json: Dict, api_key: str, model: str = "gpt-4.1-2025-04-14") -> Dict:
    """
    Synchronous wrapper for complete analysis

    Args:
        trajectory_json: Complete trajectory data (dict from JSON)
        api_key: OpenAI API key
        model: Model to use (default: gpt-4.1-2025-04-14)

    Returns:
        Dictionary containing both phase1 errors and critical error

    Example:
        results = analyze_trajectory_sync(trajectory_data, "your-api-key")
        critical = results['critical_error']
        print(f"Critical error at step {critical['critical_step']}: {critical['error_type']}")
    """
    detector = AgentErrorDetectorAPI(api_key, model)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(detector.analyze_trajectory(trajectory_json))
    finally:
        loop.close()


def detect_errors_sync(trajectory_json: Dict, api_key: str, model: str = "gpt-4.1-2025-04-14") -> Dict:
    """
    Synchronous wrapper for Phase 1 only

    Args:
        trajectory_json: Complete trajectory data
        api_key: OpenAI API key
        model: Model to use

    Returns:
        Step-by-step error analysis
    """
    detector = AgentErrorDetectorAPI(api_key, model)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(detector.detect_errors(trajectory_json))
    finally:
        loop.close()


def find_critical_error_sync(phase1_results: Dict, trajectory_json: Dict, api_key: str, model: str = "gpt-4.1-2025-04-14") -> Dict:
    """
    Synchronous wrapper for Phase 2 only

    Args:
        phase1_results: Results from Phase 1
        trajectory_json: Original trajectory data
        api_key: OpenAI API key
        model: Model to use

    Returns:
        Critical error identification
    """
    detector = AgentErrorDetectorAPI(api_key, model)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(detector.find_critical_error(phase1_results, trajectory_json))
    finally:
        loop.close()


def get_error_definitions_sync(module: Optional[str] = None) -> Dict:
    """
    Get error type definitions (no API key needed)

    Args:
        module: Specific module or None for all

    Returns:
        Error definitions dictionary
    """
    loader = ErrorDefinitionsLoader()
    if module:
        return loader.get_module_definitions(module)
    else:
        all_defs = {}
        for mod in ['memory', 'reflection', 'planning', 'action', 'system', 'others']:
            all_defs[mod] = loader.get_module_definitions(mod)
        return all_defs


# Example usage
if __name__ == "__main__":
    # Example trajectory file path
    example_file = "/Users/liuzijia/Downloads/detector/human_annotation/GPT-4o/chat_b000_t00_e00-d3248f80.json"

    # Load trajectory
    with open(example_file, 'r') as f:
        trajectory = json.load(f)

    # Example API key (replace with actual)
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    if api_key == "your-api-key-here":
        print("Please set OPENAI_API_KEY environment variable")
        print("\nAvailable error definitions:")
        defs = get_error_definitions_sync()
        for module, errors in defs.items():
            print(f"\n{module.upper()}:")
            for error_type in errors:
                print(f"  - {error_type}")
    else:
        # Analyze trajectory
        print("Analyzing trajectory...")
        results = analyze_trajectory_sync(trajectory, api_key)

        # Display results
        if results['critical_error']:
            critical = results['critical_error']
            print(f"\nCritical Error Found:")
            print(f"  Step: {critical['critical_step']}")
            print(f"  Module: {critical['critical_module']}")
            print(f"  Type: {critical['error_type']}")
            print(f"  Guidance: {critical['correction_guidance']}")
        else:
            print("\nNo critical error identified")