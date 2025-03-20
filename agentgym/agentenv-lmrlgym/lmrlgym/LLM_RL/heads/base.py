from __future__ import annotations
from typing import Dict, Any
import json

class HeadConfig:
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HeadConfig:
        return cls(**config_dict)
    
    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
