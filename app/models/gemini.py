from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class Blob(CamelBaseModel):
    mime_type: str
    data: str  # Base64 encoded


class FunctionCall(CamelBaseModel):
    name: str
    args: Dict[str, Any]


class FunctionResponse(CamelBaseModel):
    name: str
    response: Dict[str, Any]


class FileData(CamelBaseModel):
    mime_type: Optional[str] = None
    file_uri: str


class Part(CamelBaseModel):
    text: Optional[str] = None
    inline_data: Optional[Blob] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[FunctionResponse] = None
    file_data: Optional[FileData] = None


class Content(CamelBaseModel):
    parts: List[Part]
    role: Optional[str] = None


class GenerationConfig(CamelBaseModel):
    candidate_count: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    response_mime_type: Optional[str] = None
    response_schema: Optional[Dict[str, Any]] = None


class SafetySetting(CamelBaseModel):
    category: str
    threshold: str


class Tool(CamelBaseModel):
    function_declarations: Optional[List[Dict[str, Any]]] = None
    google_search_retrieval: Optional[Dict[str, Any]] = None
    code_execution: Optional[Dict[str, Any]] = None


class GenerateContentRequest(CamelBaseModel):
    contents: List[Content]
    tools: Optional[List[Tool]] = None
    safety_settings: Optional[List[SafetySetting]] = None
    generation_config: Optional[GenerationConfig] = None
    system_instruction: Optional[Content] = None


class SafetyRating(CamelBaseModel):
    category: str
    probability: str
    blocked: Optional[bool] = None


class CitationSource(CamelBaseModel):
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    uri: Optional[str] = None
    license: Optional[str] = None


class CitationMetadata(CamelBaseModel):
    citation_sources: List[CitationSource]


class Candidate(CamelBaseModel):
    content: Optional[Content] = None
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[SafetyRating]] = None
    citation_metadata: Optional[CitationMetadata] = None
    token_count: Optional[int] = None
    index: Optional[int] = None


class PromptFeedback(CamelBaseModel):
    block_reason: Optional[str] = None
    safety_ratings: Optional[List[SafetyRating]] = None


class UsageMetadata(CamelBaseModel):
    prompt_token_count: int
    candidates_token_count: Optional[int] = None
    total_token_count: int


class GenerateContentResponse(CamelBaseModel):
    candidates: List[Candidate]
    prompt_feedback: Optional[PromptFeedback] = None
    usage_metadata: Optional[UsageMetadata] = None


class Model(CamelBaseModel):
    name: str
    base_model_id: Optional[str] = None
    version: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    supported_generation_methods: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


class ModelList(CamelBaseModel):
    models: List[Model]
