import json
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from gemini_webapi.constants import Model as GeminiModel
from loguru import logger

from app.models.gemini import (
    Blob,
    Candidate,
    Content,
    GenerateContentRequest,
    GenerateContentResponse,
    Model,
    ModelList,
    Part,
    UsageMetadata,
)
from app.models.models import ContentItem, Message
from app.server.common import get_model_by_name, image_to_base64
from app.server.middleware import get_image_store_dir, get_temp_dir
from app.services.client import GeminiClientWrapper
from app.services.pool import GeminiClientPool
from app.utils import g_config
from app.utils.helper import estimate_tokens, save_file_to_tempfile

router = APIRouter()


async def verify_gemini_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
    x_goog_api_key: str | None = Header(default=None, alias="x-goog-api-key"),
    key: str | None = Query(default=None),
) -> str:
    """
    Verify API key from Bearer token, x-goog-api-key header, or 'key' query parameter.
    """
    expected_key = g_config.server.api_key
    if not expected_key:
        return ""

    # 1. Check Bearer Token
    if credentials and credentials.scheme.lower() == "bearer":
        if credentials.credentials == expected_key:
            return credentials.credentials

    # 2. Check x-goog-api-key Header
    if x_goog_api_key and x_goog_api_key == expected_key:
        return x_goog_api_key

    # 3. Check 'key' Query Parameter
    if key and key == expected_key:
        return key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key. Please provide it via 'Authorization: Bearer', 'x-goog-api-key' header, or 'key' query parameter."
    )


def _convert_gemini_content_to_message(content: Content) -> Message:
    """Convert Gemini API Content to internal Message model."""
    role = content.role or "user"
    if role == "model":
        role = "assistant"

    parts = []
    for part in content.parts:
        if part.text:
            parts.append(ContentItem(type="text", text=part.text))
        elif part.inline_data:
            # We will handle inline_data by saving it later or converting to appropriate format
            # For now, let's keep it as is or mark it for processing
            # Internal Message expects image_url or file
            # We'll need to save base64 to temp file
            pass

    # This function needs to be async to save files, or we handle files outside
    # Let's handle files outside or use a placeholder
    return Message(role=role, content=parts if parts else None)


async def _process_gemini_request(
    request: GenerateContentRequest, temp_dir: Path
) -> tuple[List[Message], List[Path]]:
    """Convert Gemini request to messages and files."""
    messages = []
    files = []

    for content in request.contents:
        role = content.role or "user"
        if role == "model":
            role = "assistant"

        message_parts = []

        for part in content.parts:
            if part.text:
                message_parts.append(ContentItem(type="text", text=part.text))

            if part.inline_data:
                # Save base64 image to temp file
                try:
                    file_path = await save_file_to_tempfile(
                        part.inline_data.data,
                        file_name=f"img_{uuid.uuid4().hex}.{part.inline_data.mime_type.split('/')[-1]}",
                        tempdir=temp_dir
                    )
                    files.append(file_path)
                    # We don't add image to message content for gemini-webapi usually,
                    # it takes files separately in send_message.
                    # But for history tracking, maybe we should?
                    # The current implementation of process_message extracts files from Message.
                    # Here we extract directly.
                    message_parts.append(ContentItem(type="text", text="[Image]"))
                except Exception as e:
                    logger.error(f"Failed to process inline image: {e}")

            if part.file_data:
                # Handle file URI if needed, currently not supported by helper
                logger.warning(f"file_data not fully supported: {part.file_data}")

        messages.append(Message(role=role, content=message_parts if message_parts else ""))

    return messages, files


@router.get("/v1beta/models", response_model=ModelList)
async def list_models(
    api_key: str = Depends(verify_gemini_api_key),
):
    """List available models."""
    models = []
    for m in GeminiModel:
        if m.name == "UNSPECIFIED":
            continue

        # Use the model_name property which is correctly populated by __init__
        model_id = m.model_name

        # Using mock values for limits as they are not provided by the library
        models.append(
            Model(
                name=f"models/{model_id}",
                base_model_id=model_id,
                version="001",
                display_name=model_id.replace("-", " ").title(),
                description=f"Google Gemini model {model_id}",
                input_token_limit=32768,
                output_token_limit=8192,
                supported_generation_methods=["generateContent", "countTokens"],
            )
        )
    return ModelList(models=models)


@router.get("/v1beta/models/{model}", response_model=Model)
async def get_model(
    model: str,
    api_key: str = Depends(verify_gemini_api_key),
):
    """Get information about a specific model."""
    model_name = model.replace("models/", "")

    # Check if model exists in GeminiModel enum
    found_model = None
    for m in GeminiModel:
        if m.name == "UNSPECIFIED":
            continue

        if m.model_name == model_name:
            found_model = m
            break

    if not found_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model} not found")

    model_id = found_model.model_name
    return Model(
        name=f"models/{model_id}",
        base_model_id=model_id,
        version="001",
        display_name=model_id.replace("-", " ").title(),
        description=f"Google Gemini model {model_id}",
        input_token_limit=32768,
        output_token_limit=8192,
        supported_generation_methods=["generateContent", "countTokens"],
    )


@router.post("/v1beta/models/{model}:generateContent", response_model=GenerateContentResponse)
async def generate_content(
    model: str,
    request: GenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    temp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
    # Fix model name if needed (remove models/ prefix)
    model_name = model.replace("models/", "")

    try:
        model_obj = get_model_by_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    messages, files = await _process_gemini_request(request, temp_dir)

    if not messages:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content provided.")

    # Use GeminiClientWrapper to process conversation history into a single prompt
    input_text, all_files = await GeminiClientWrapper.process_conversation(messages, temp_dir)

    # If process_conversation handles files, we might have duplicates if we also use files from _process_gemini_request
    # actually _process_gemini_request returns files that were newly created from inline_data
    # GeminiClientWrapper.process_conversation also extracts files from messages
    # But since we constructed messages in _process_gemini_request with local file paths (if we did correctly),
    # wait, _process_gemini_request adds ContentItem(type="text", text="[Image]") for inline images
    # and returns files separately.
    # GeminiClientWrapper.process_conversation expects messages to have image_url or file_data for files.
    # So my previous implementation of _process_gemini_request was slightly incompatible with process_conversation if I want it to handle files.
    # However, _process_gemini_request ALREADY downloaded images to files.
    # So I should just append those files to whatever process_conversation finds (which should be none if I only used text).

    # Let's adjust _process_gemini_request to NOT put [Image] text but maybe something else or just rely on the files list.
    # Actually, GeminiWebAPI (the library) takes text and files.
    # The wrapper's process_conversation merges text.

    # Let's check _process_gemini_request again.
    # It appends ContentItem(type="text", text="[Image]")

    pool = GeminiClientPool()

    try:
        client = await pool.acquire()
        session = client.start_chat(model=model_obj)

        # Combine files from request and any potential files from conversation processing (though likely empty here)
        # But wait, all_files from process_conversation comes from message.content items of type 'file' or 'image_url'.
        # In _process_gemini_request, I only created text items.
        # So all_files will be empty (except for the files list I got from _process_gemini_request).

        final_files = files + all_files

        response = await session.send_message(input_text, files=final_files)

    except Exception as e:
        logger.exception("Gemini API error")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

    # Process response
    candidates = []

    response_parts = []

    # Text
    if response.text:
        response_parts.append(Part(text=response.text))

    # Images
    if response.images:
        for img in response.images:
            try:
                # Convert to base64
                b64_str, _, _, _, _ = await image_to_base64(img, image_store)
                response_parts.append(
                    Part(
                        inline_data=Blob(
                            mime_type="image/jpeg", # defaulting to jpeg, or detect
                            data=b64_str
                        )
                    )
                )
            except Exception as e:
                logger.error(f"Failed to process response image: {e}")

    candidates.append(
        Candidate(
            content=Content(role="model", parts=response_parts),
            finish_reason="STOP",
            index=0
        )
    )

    # Usage
    prompt_tokens = estimate_tokens(input_text)
    completion_tokens = estimate_tokens(response.text)

    return GenerateContentResponse(
        candidates=candidates,
        usage_metadata=UsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens
        )
    )


@router.post("/v1beta/models/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    request: GenerateContentRequest,
    api_key: str = Depends(verify_gemini_api_key),
    temp_dir: Path = Depends(get_temp_dir),
    alt: str | None = Query(default=None),
):
    """
    Generates a streaming response from the model given an input.
    """
    model_name = model.replace("models/", "")

    # Validate model
    try:
        model_obj = get_model_by_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    messages, files = await _process_gemini_request(request, temp_dir)

    if not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content provided.")

    # Process conversation history
    input_text, all_files = await GeminiClientWrapper.process_conversation(messages, temp_dir)

    final_files = files + all_files

    pool = GeminiClientPool()
    try:
        client = await pool.acquire()
        session = client.start_chat(model=model_obj)
    except Exception as e:
        logger.exception("Failed to acquire Gemini client")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Failed to connect to Gemini: {e!s}")

    is_sse = (alt == "sse")

    async def generate_chunks():
        try:
            if not is_sse:
                yield "["

            first = True
            previous_text = ""

            # Keep track for usage
            prompt_tokens = estimate_tokens(input_text)

            async for chunk in session.send_message_stream(input_text, files=final_files):
                current_text = chunk.text
                delta = ""
                if current_text.startswith(previous_text):
                    delta = current_text[len(previous_text):]
                    previous_text = current_text
                else:
                    # Fallback if it doesn't start with previous (shouldn't happen usually)
                    delta = current_text
                    previous_text = current_text

                if delta:
                    if not is_sse and not first:
                        yield ","
                    first = False

                    response_data = GenerateContentResponse(
                        candidates=[
                            Candidate(
                                content=Content(
                                    parts=[Part(text=delta)],
                                    role="model"
                                ),
                                finish_reason=None,
                                index=0,
                                safety_ratings=[]
                            )
                        ]
                    )

                    json_str = json.dumps(response_data.model_dump(mode="json", exclude_none=True, by_alias=True))
                    if is_sse:
                        yield f"data: {json_str}\n\n"
                    else:
                        yield json_str

            # Send one final chunk with finish reason and usage
            if not is_sse and not first:
                yield ","

            completion_tokens = estimate_tokens(previous_text)

            final_response = GenerateContentResponse(
                candidates=[
                    Candidate(
                        content=Content(parts=[], role="model"),
                        finish_reason="STOP",
                        index=0,
                        safety_ratings=[]
                    )
                ],
                usage_metadata=UsageMetadata(
                    prompt_token_count=prompt_tokens,
                    candidates_token_count=completion_tokens,
                    total_token_count=prompt_tokens + completion_tokens
                )
            )

            json_str = json.dumps(final_response.model_dump(mode="json", exclude_none=True, by_alias=True))
            if is_sse:
                yield f"data: {json_str}\n\n"
            else:
                yield json_str

            if not is_sse:
                yield "]"

        except Exception:
            logger.exception("Error in stream_generate_content")
            # If we already started streaming, we can't cleanly error out with HTTP 500.
            # We might yield a JSON error object if the client supports it, or just stop.
            pass

    return StreamingResponse(generate_chunks(), media_type="text/event-stream" if is_sse else "application/json")


@router.post("/v1beta/models/{model}:countTokens")
async def count_tokens(
    model: str,
    request: GenerateContentRequest,
    api_key: str = Depends(verify_gemini_api_key),
    temp_dir: Path = Depends(get_temp_dir),
):
    """
    Counts tokens for the input.
    """
    model_name = model.replace("models/", "")
    try:
        _ = get_model_by_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    messages, _ = await _process_gemini_request(request, temp_dir)
    if not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content provided.")

    input_text, _ = await GeminiClientWrapper.process_conversation(messages, temp_dir)

    count = estimate_tokens(input_text)

    return {"totalTokens": count}
