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
from analysis_error_detection import ErrorTypeDetector
from analysis_phase2_critical import CriticalErrorAnalyzer
from error_definitions_loader import ErrorDefinitionsLoader


class AgentErrorDetectorAPI:
    """
    Main API class for error detection in agent trajectories

    This class provides a simple interface for external systems to:
    1. Detect all errors in a failed trajectory (Phase 1)
    2. Identify the critical error that caused failure (Phase 2)
    3. Get correction guidance for trajectory reload
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-2025-04-14",
        capture_debug_data: bool = False,
        base_url: Optional[str] = None,
        phase1_parallel_workers: int = 1,
    ):
        """
        Initialize the error detector.

        Args:
            api_key: OpenAI API key (can be empty for local vLLM without auth)
            model: Model to use for analysis
            capture_debug_data: Whether to capture debug payloads
            base_url: Optional OpenAI-compatible base URL. If provided, it should be the
                API base like "http://host:port/v1". The detector will post to the
                chat completions path under it. If not provided, defaults to OpenAI.
        """

        # Normalize base URL to the full chat completions endpoint
        if base_url:
            url = base_url.rstrip('/')
            # Ensure it points to /v1/chat/completions
            if url.endswith('/chat/completions'):
                normalized = url
            elif url.endswith('/v1'):
                normalized = f"{url}/chat/completions"
            else:
                # Assume caller passed bare host:port without /v1; append /v1/chat/completions
                normalized = f"{url}/v1/chat/completions"
        else:
            normalized = "https://api.openai.com/v1/chat/completions"

        self.api_config = {
            "base_url": normalized,
            "api_key": api_key,
            "model": model,
            "temperature": 0.0,
            "max_retries": 3,
            "timeout": 60,
        }

        self.capture_debug_data = capture_debug_data
        self.phase1_parallel_workers = max(1, int(phase1_parallel_workers))
        self.phase1_detector = ErrorTypeDetector(self.api_config, parallel_workers=self.phase1_parallel_workers)
        self.phase2_analyzer = CriticalErrorAnalyzer(self.api_config, capture_debug_data=capture_debug_data)
        self.error_loader = ErrorDefinitionsLoader()

    async def analyze_trajectory(
        self,
        trajectory_json: Dict,
        *,
        previous_phase1: Optional[Dict[str, Any]] = None,
        attempt_index: int = 1,
        recompute_from_step: Optional[int] = None,
    ) -> Dict:
        """Complete analysis pipeline: Phase 1 + Phase 2 with optional caching."""

        cached_step_analyses: Optional[List[Dict[str, Any]]] = None
        if previous_phase1:
            cached_step_analyses = previous_phase1.get('step_analyses')

        # Default recompute window to 1 (start from first step) if not provided
        recompute = recompute_from_step or 1

        # Phase 1: Detect all errors (optionally reusing cached analyses)
        phase1_results = await self.detect_errors(
            trajectory_json,
            recompute_from_step=recompute,
            cached_step_analyses=cached_step_analyses,
        )

        # Phase 2: Find critical error with iterative guidance context
        phase2_results = await self.find_critical_error(
            phase1_results,
            trajectory_json,
            attempt_index=attempt_index,
        )

        critical_error = phase2_results.get('critical_error') if isinstance(phase2_results, dict) else None
        follow_up_instruction = None
        if isinstance(critical_error, dict):
            follow_up_instruction = critical_error.get('follow_up_instruction')

        result = {
            'phase1_errors': phase1_results,
            'critical_error': critical_error,
            'follow_up_instruction': follow_up_instruction,
            'task_success': phase1_results.get('task_success', False),
            'environment': phase1_results.get('environment', 'unknown')
        }

        if self.capture_debug_data and isinstance(phase2_results, dict) and phase2_results.get('debug_payload'):
            result['debug_payload'] = phase2_results['debug_payload']

        return result

    async def detect_errors(
        self,
        trajectory_json: Dict,
        *,
        recompute_from_step: int = 1,
        cached_step_analyses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        """Phase 1: Detect all errors in trajectory."""
        trajectory_data = self.phase1_detector.parse_trajectory_from_dict(trajectory_json)
        results = await self.phase1_detector.analyze_trajectory(
            trajectory_data,
            recompute_from_step=recompute_from_step,
            cached_step_analyses=cached_step_analyses,
        )
        return results

    async def find_critical_error(
        self,
        phase1_results: Dict,
        trajectory_json: Dict,
        *,
        attempt_index: int = 1,
    ) -> Dict:
        """Phase 2: Identify critical failure point."""
        critical_error = await self.phase2_analyzer.identify_critical_error(
            phase1_results,
            trajectory_json,
            attempt_index=attempt_index,
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
def analyze_trajectory_sync(
    trajectory_json: Dict,
    api_key: str,
    model: str = "gpt-4.1-2025-04-14",
    base_url: Optional[str] = None,
) -> Dict:
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
    detector = AgentErrorDetectorAPI(api_key, model, base_url=base_url)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(detector.analyze_trajectory(trajectory_json))
    finally:
        loop.close()


def detect_errors_sync(
    trajectory_json: Dict,
    api_key: str,
    model: str = "gpt-4.1-2025-04-14",
    base_url: Optional[str] = None,
) -> Dict:
    """
    Synchronous wrapper for Phase 1 only

    Args:
        trajectory_json: Complete trajectory data
        api_key: OpenAI API key
        model: Model to use

    Returns:
        Step-by-step error analysis
    """
    detector = AgentErrorDetectorAPI(api_key, model, base_url=base_url)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(detector.detect_errors(trajectory_json))
    finally:
        loop.close()


def find_critical_error_sync(
    phase1_results: Dict,
    trajectory_json: Dict,
    api_key: str,
    model: str = "gpt-4.1-2025-04-14",
    base_url: Optional[str] = None,
) -> Dict:
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
    detector = AgentErrorDetectorAPI(api_key, model, base_url=base_url)
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
