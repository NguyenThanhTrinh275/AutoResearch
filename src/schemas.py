from pydantic import BaseModel, Field
from typing import List

class PlannerOutput(BaseModel):
    queries: List[str] = Field(
        description="The list should include 2 to 3 short search queries, focusing on different aspects of the topic."
    )

class DocumentGrade(BaseModel):
    binary_score: str = Field(
        description="Does the document actually contain information useful to the topic? Just return 'yes' or 'no'."
    )
    
class CriticSchema(BaseModel):
    is_perfect: bool = Field(
        description="Set to True if the draft is flawless and strictly backed by the context. Set to False if it requires edits."
    )
    feedback: str = Field(
        description="Detailed constructive feedback pointing out specific errors, hallucinations, or missing points. Leave empty if is_perfect is True."
    )