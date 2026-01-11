from typing import Dict, Optional

from pydantic import BaseModel


class HealthCheckRs(BaseModel):
    status: str
    version: Optional[str] = None
    output: Optional[str] = None
    details: Optional[Dict] = None
