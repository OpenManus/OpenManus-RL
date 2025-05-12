from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple, Union


# API Models
class CreateQuery(BaseModel):
    """Create environment request"""

    id: int = 0
    dataset_type: str = "validation"
    tool_list: Optional[List[str]] = None


class StepQuery(BaseModel):
    """Execute action request"""

    env_idx: str  # Changed to string for UUID
    action: str


class StepResponse(BaseModel):
    """Execute action response"""

    observation: str
    reward: float
    done: bool
    info: dict


class ResetQuery(BaseModel):
    """Reset environment request"""

    env_idx: str  # Changed to string for UUID
    id: Optional[int] = None
    dataset_type: Optional[str] = "validation"
