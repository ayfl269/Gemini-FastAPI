from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ContentItem(BaseModel):
    """Individual content item (text, image, or file) within a message."""

    type: Literal["text", "image_url", "file", "input_audio"]
    text: str | None = Field(default=None)
    image_url: dict[str, Any] | None = Field(default=None)
    input_audio: dict[str, Any] | None = Field(default=None)
    file: dict[str, Any] | None = Field(default=None)
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class Message(BaseModel):
    """Message model"""

    role: str
    content: str | list[ContentItem] | None = Field(default=None)
    name: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)
    refusal: str | None = Field(default=None)
    reasoning_content: str | None = Field(default=None)
    audio: dict[str, Any] | None = Field(default=None)
    annotations: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_role(self) -> Message:
        """Normalize 'developer' role to 'system' for Gemini compatibility."""
        if self.role == "developer":
            self.role = "system"
        return self


class Choice(BaseModel):
    """Choice model"""

    index: int
    message: Message
    finish_reason: str
    logprobs: dict[str, Any] | None = Field(default=None)


class FunctionCall(BaseModel):
    """Function call payload"""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call item"""

    id: str
    type: Literal["function"]
    function: FunctionCall


class ToolFunctionDefinition(BaseModel):
    """Function definition for tool."""

    name: str
    description: str | None = Field(default=None)
    parameters: dict[str, Any] | None = Field(default=None)


class Tool(BaseModel):
    """Tool specification."""

    type: Literal["function"]
    function: ToolFunctionDefinition


class ToolChoiceFunctionDetail(BaseModel):
    """Detail of a tool choice function."""

    name: str


class ToolChoiceFunction(BaseModel):
    """Tool choice forcing a specific function."""

    type: Literal["function"]
    function: ToolChoiceFunctionDetail


class Usage(BaseModel):
    """Usage statistics model"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int] | None = Field(default=None)
    completion_tokens_details: dict[str, int] | None = Field(default=None)


class ModelData(BaseModel):
    """Model data model"""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""

    model: str
    messages: list[Message]
    stream: bool | None = Field(default=False)
    user: str | None = Field(default=None)
    temperature: float | None = Field(default=0.7)
    top_p: float | None = Field(default=1.0)
    max_tokens: int | None = Field(default=None)
    tools: list[Tool] | None = Field(default=None)
    tool_choice: (
        Literal["none"] | Literal["auto"] | Literal["required"] | ToolChoiceFunction | None
    ) = Field(default=None)
    response_format: dict[str, Any] | None = Field(default=None)


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelListResponse(BaseModel):
    """Model list model"""

    object: str = "list"
    data: list[ModelData]


class HealthCheckResponse(BaseModel):
    """Health check response model"""

    ok: bool
    storage: dict[str, Any] | None = Field(default=None)
    clients: dict[str, bool] | None = Field(default=None)
    error: str | None = Field(default=None)


class ConversationInStore(BaseModel):
    """Conversation model for storing in the database."""

    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)

    # Gemini Web API does not support changing models once a conversation is created.
    model: str = Field(..., description="Model used for the conversation")
    client_id: str = Field(..., description="Identifier of the Gemini client")
    metadata: list[str | None] = Field(
        ..., description="Metadata for Gemini API to locate the conversation"
    )
    messages: list[Message] = Field(..., description="Message contents in the conversation")


class ResponseInputContent(BaseModel):
    """Content item for Responses API input."""

    type: Literal["input_text", "output_text", "reasoning_text", "input_image", "input_file"]
    text: str | None = Field(default=None)
    image_url: str | None = Field(default=None)
    detail: Literal["auto", "low", "high"] | None = Field(default=None)
    file_url: str | None = Field(default=None)
    file_data: str | None = Field(default=None)
    filename: str | None = Field(default=None)
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseInputItem(BaseModel):
    """Single input item for Responses API."""

    type: Literal["message"] | None = Field(default="message")
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[ResponseInputContent]


class ResponseToolChoice(BaseModel):
    """Tool choice enforcing a specific tool in Responses API."""

    type: Literal["function", "image_generation"]
    function: ToolChoiceFunctionDetail | None = Field(default=None)


class ResponseImageTool(BaseModel):
    """Image generation tool specification for Responses API."""

    type: Literal["image_generation"]
    model: str | None = Field(default=None)
    output_format: str | None = Field(default=None)


class ResponseCreateRequest(BaseModel):
    """Responses API request payload."""

    model: str
    input: str | list[ResponseInputItem]
    instructions: str | list[ResponseInputItem] | None = Field(default=None)
    temperature: float | None = Field(default=0.7)
    top_p: float | None = Field(default=1.0)
    max_output_tokens: int | None = Field(default=None)
    stream: bool | None = Field(default=False)
    tool_choice: str | ResponseToolChoice | None = Field(default=None)
    tools: list[Tool | ResponseImageTool] | None = Field(default=None)
    store: bool | None = Field(default=None)
    user: str | None = Field(default=None)
    response_format: dict[str, Any] | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class ResponseUsage(BaseModel):
    """Usage statistics for Responses API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, Any] = Field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens_details: dict[str, Any] = Field(default_factory=lambda: {"reasoning_tokens": 0})


class ResponseOutputContent(BaseModel):
    """Content item for Responses API output."""

    type: Literal["output_text"]
    text: str | None = Field(default="")
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = Field(default=None)


class ResponseOutputMessage(BaseModel):
    """Assistant message returned by Responses API."""

    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed", "incomplete"] = Field(default="completed")
    role: Literal["assistant"]
    content: list[ResponseOutputContent]


class ResponseSummaryPart(BaseModel):
    """Summary part for reasoning."""

    type: Literal["summary_text"] = Field(default="summary_text")
    text: str


class ResponseReasoningContentPart(BaseModel):
    """Content part for reasoning."""

    type: Literal["reasoning_text"] = Field(default="reasoning_text")
    text: str


class ResponseReasoning(BaseModel):
    """Reasoning item returned by Responses API."""

    id: str
    type: Literal["reasoning"] = Field(default="reasoning")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(default=None)
    summary: list[ResponseSummaryPart] | None = Field(default=None)
    content: list[ResponseReasoningContentPart] | None = Field(default=None)


class ResponseImageGenerationCall(BaseModel):
    """Image generation call record emitted in Responses API."""

    id: str
    type: Literal["image_generation_call"] = Field(default="image_generation_call")
    status: Literal["completed", "in_progress", "generating", "failed"] = Field(default="completed")
    result: str | None = Field(default=None)
    output_format: str | None = Field(default=None)
    size: str | None = Field(default=None)
    revised_prompt: str | None = Field(default=None)


class ResponseToolCall(BaseModel):
    """Tool call record emitted in Responses API."""

    id: str
    type: Literal["tool_call"] = Field(default="tool_call")
    status: Literal["in_progress", "completed", "failed", "requires_action"] = Field(
        default="completed"
    )
    function: FunctionCall


class ResponseTextFormat(BaseModel):
    """Text format configuration for Responses API."""

    type: Literal["text", "json_schema"] = Field(default="text")


class ResponseTextConfig(BaseModel):
    """Text configuration for Responses API."""

    format: ResponseTextFormat = Field(default_factory=ResponseTextFormat)


class ResponseCreateResponse(BaseModel):
    """Responses API response payload."""

    id: str
    object: Literal["response"] = Field(default="response")
    created_at: int
    completed_at: int | None = Field(default=None)
    model: str
    output: list[
        ResponseReasoning | ResponseOutputMessage | ResponseImageGenerationCall | ResponseToolCall
    ]
    status: Literal[
        "in_progress",
        "completed",
        "failed",
        "incomplete",
        "cancelled",
        "requires_action",
    ] = Field(default="completed")
    tool_choice: str | ToolChoiceFunction | ResponseToolChoice = Field(default="auto")
    tools: list[Tool | ResponseImageTool] = Field(default_factory=list)
    usage: ResponseUsage | None = Field(default=None)
    error: dict[str, Any] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    input: str | list[ResponseInputItem] | None = Field(default=None)
    text: ResponseTextConfig | None = Field(default_factory=ResponseTextConfig)


# Rebuild models with forward references
Message.model_rebuild()
ToolCall.model_rebuild()
ChatCompletionRequest.model_rebuild()


# ============== Google Gemini Native API Models ==============


class GeminiPart(BaseModel):
    """Content part in Gemini API format."""

    text: str | None = Field(default=None)
    inline_data: dict[str, Any] | None = Field(default=None, alias="inlineData")
    file_data: dict[str, Any] | None = Field(default=None, alias="fileData")
    function_call: dict[str, Any] | None = Field(default=None, alias="functionCall")
    function_response: dict[str, Any] | None = Field(default=None, alias="functionResponse")

    model_config = {"populate_by_name": True}


class GeminiContent(BaseModel):
    """Content object in Gemini API format."""

    role: Literal["user", "model", "function", "system"]
    parts: list[GeminiPart]


class GeminiGenerationConfig(BaseModel):
    """Generation configuration for Gemini API."""

    temperature: float | None = Field(default=None)
    top_p: float | None = Field(default=None, alias="topP")
    top_k: int | None = Field(default=None, alias="topK")
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    candidate_count: int | None = Field(default=1, alias="candidateCount")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")
    response_mime_type: str | None = Field(default=None, alias="responseMimeType")
    response_schema: dict[str, Any] | None = Field(default=None, alias="responseSchema")
    presence_penalty: float | None = Field(default=None, alias="presencePenalty")
    frequency_penalty: float | None = Field(default=None, alias="frequencyPenalty")
    seed: int | None = Field(default=None)
    thinking_config: dict[str, Any] | None = Field(default=None, alias="thinkingConfig")

    model_config = {"populate_by_name": True}


class GeminiFunctionDeclaration(BaseModel):
    """Function declaration for Gemini tools."""

    name: str
    description: str | None = Field(default=None)
    parameters: dict[str, Any] | None = Field(default=None)


class GeminiTool(BaseModel):
    """Tool specification in Gemini API format."""

    function_declarations: list[GeminiFunctionDeclaration] | None = Field(
        default=None, alias="functionDeclarations"
    )
    code_execution: dict[str, Any] | None = Field(default=None, alias="codeExecution")

    model_config = {"populate_by_name": True}


class GeminiSafetySetting(BaseModel):
    """Safety setting in Gemini API format."""

    category: str
    threshold: str


class GeminiGenerateContentRequest(BaseModel):
    """Request body for generateContent endpoint."""

    contents: list[GeminiContent]
    system_instruction: GeminiContent | dict[str, Any] | None = Field(
        default=None, alias="systemInstruction"
    )
    generation_config: GeminiGenerationConfig | None = Field(
        default=None, alias="generationConfig"
    )
    safety_settings: list[GeminiSafetySetting] | None = Field(
        default=None, alias="safetySettings"
    )
    tools: list[GeminiTool] | None = Field(default=None)

    model_config = {"populate_by_name": True}


class GeminiCandidate(BaseModel):
    """Candidate response from Gemini API."""

    content: GeminiContent | None = Field(default=None)
    finish_reason: str | None = Field(default=None, alias="finishReason")
    safety_ratings: list[dict[str, Any]] | None = Field(default=None, alias="safetyRatings")
    citation_metadata: dict[str, Any] | None = Field(default=None, alias="citationMetadata")
    token_count: int | None = Field(default=None, alias="tokenCount")
    index: int | None = Field(default=0)

    model_config = {"populate_by_name": True}


class GeminiUsageMetadata(BaseModel):
    """Usage metadata from Gemini API."""

    prompt_token_count: int | None = Field(default=None, alias="promptTokenCount")
    candidates_token_count: int | None = Field(default=None, alias="candidatesTokenCount")
    total_token_count: int | None = Field(default=None, alias="totalTokenCount")

    model_config = {"populate_by_name": True}


class GeminiGenerateContentResponse(BaseModel):
    """Response body for generateContent endpoint."""

    candidates: list[GeminiCandidate] | None = Field(default=None)
    usage_metadata: GeminiUsageMetadata | None = Field(
        default=None, alias="usageMetadata"
    )
    prompt_feedback: dict[str, Any] | None = Field(default=None, alias="promptFeedback")
    model_version: str | None = Field(default=None, alias="modelVersion")

    model_config = {"populate_by_name": True}


class GeminiModelInfo(BaseModel):
    """Model information for list models response."""

    name: str
    version: str | None = Field(default=None)
    display_name: str | None = Field(default=None, alias="displayName")
    description: str | None = Field(default=None)
    input_token_limit: int | None = Field(default=None, alias="inputTokenLimit")
    output_token_limit: int | None = Field(default=None, alias="outputTokenLimit")
    supported_generation_methods: list[str] | None = Field(
        default=None, alias="supportedGenerationMethods"
    )

    model_config = {"populate_by_name": True}


class GeminiListModelsResponse(BaseModel):
    """Response for listing available models."""

    models: list[GeminiModelInfo] | None = Field(default=None)
    next_page_token: str | None = Field(default=None, alias="nextPageToken")

    model_config = {"populate_by_name": True}
