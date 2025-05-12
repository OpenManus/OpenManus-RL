"""
OpenManus Tree of Thought Agent Module

This module extends the OpenManus agent with Tree of Thought capabilities
for more effective exploration during reinforcement learning.
"""

from .openmanus_tot import OpenManusToTAgent, ToTConfig

__all__ = ['OpenManusToTAgent', 'ToTConfig']