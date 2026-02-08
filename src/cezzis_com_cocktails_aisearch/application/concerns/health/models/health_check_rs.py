from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class HealthCheckRs(BaseModel):
    status: str
    version: Optional[str] = None
    output: Optional[str] = None
    details: Optional[Dict] = None

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "output": "API is running",
                "details": {
                    "qdrant_database": "healthy",
                },
            }
        },
    )
