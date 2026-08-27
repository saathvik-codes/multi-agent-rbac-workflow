from pydantic import BaseModel, Field
from typing import Optional

class TaskRequest(BaseModel):
    prompt: str = Field(..., description="The query input to run through the multi-agent system")
    user_role: str = Field("Public", description="The clearance role of the user executing the query")

class TaskResponse(BaseModel):
    task_id: str
    prompt: str
    user_role: str
    response: str
    security_blocked: bool
    security_warnings: list[str]
    history: list[dict]
    context: list[dict]

class DocumentUploadRequest(BaseModel):
    text: str = Field(..., description="The context document chunk content")
    required_role: str = Field("Public", description="Security role requirement to view this document")
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata dictionary for the document")
