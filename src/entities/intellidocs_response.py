from pydantic import BaseModel


class IntellidocsResponse(BaseModel):
    structured_response: str
