from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict


class HealthCheckRs(BaseModel):
    status: str
    version: Optional[str] = None
    output: Optional[str] = None
    details: Optional[Dict] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "output": "API is running",
                "details": {
                    "qdrant_database": "healthy",
                },
            }
        }
    )
