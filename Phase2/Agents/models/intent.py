from typing import Dict, Any, Optional

class Intent:
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name =name
        self.params = params if params is not None else {}