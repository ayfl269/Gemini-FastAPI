import json
import re
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
        text_content = response.text
        # Filter out raw image URLs if they appear in the text
        # Only filter if we actually have images to show, otherwise keep the link
        if response.images:
            text_content = re.sub(r'https?://googleusercontent\.com/\S*', '', text_content)
            text_content = re.sub(r'http+://googleusercontent\.com/\S*', '', text_content)

        if text_content.strip():
            response_parts.append(Part(text=text_content))

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
    image_store: Path = Depends(get_image_store_dir),
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
            sent_image_count = 0

            # Buffer for handling split URLs
            text_buffer = ""
            pending_urls = []

            # Constants for URL filtering
            TARGET_URL_PREFIX = "https://googleusercontent.com"
            TARGET_URL_PREFIX_HTTP = "http://googleusercontent.com"

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

                # Append delta to buffer
                text_buffer += delta

                parts = []
                to_yield = ""

                # Robust URL extraction and buffering logic
                # We want to identify and remove Google User Content URLs, while yielding surrounding text.
                # We must be careful not to split a URL or yield a partial URL.

                # 1. Search for complete URLs in the buffer
                # Allowed URL chars (excluding CJK to fix issue with Chinese text)
                # Alphanumeric + -._~:/?#[]@!$&'()*+,;=%
                url_char_pattern = r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"

                # We loop to find all complete URLs in the current buffer
                while True:
                    # Search for start
                    match_https = re.search(r'https?://googleusercontent\.com', text_buffer)
                    match_http = re.search(r'http://googleusercontent\.com', text_buffer)

                    match = match_https or match_http
                    if match_http and match_https:
                         # Pick the earlier one
                         if match_http.start() < match_https.start():
                             match = match_http
                         else:
                             match = match_https

                    if match:
                        start_idx = match.start()
                        # Check if we have the end of this URL (followed by non-url char or end of string)
                        # We try to match as many URL chars as possible
                        rest_of_buffer = text_buffer[start_idx:]
                        full_match = re.match(r'(https?://googleusercontent\.com' + url_char_pattern + r'*)', rest_of_buffer)

                        if full_match:
                            url_end_in_rest = full_match.end()
                            url_end_abs = start_idx + url_end_in_rest

                            # If the match goes to the very end of the buffer, we are not sure if it's finished.
                            if url_end_abs < len(text_buffer):
                                # It is finished (followed by something else)
                                extracted_url = text_buffer[start_idx:url_end_abs]
                                pending_urls.append(extracted_url)

                                # We can yield everything before the URL
                                to_yield += text_buffer[:start_idx]
                                # Remove URL from buffer
                                text_buffer = text_buffer[url_end_abs:]
                                # Continue loop to find more URLs in the remaining buffer
                                continue
                            else:
                                # URL is at the end. Incomplete.
                                # Yield everything before it.
                                to_yield += text_buffer[:start_idx]
                                text_buffer = text_buffer[start_idx:]
                                break
                        else:
                            # Should not happen if regex matches start
                            break
                    else:
                        # No URL start found
                        break

                # 2. Check for partial URL start at the end of buffer
                # If the buffer ends with "http", "https://g", etc., we must wait.
                suffix_len = min(len(text_buffer), 30)
                suffix = text_buffer[-suffix_len:] if suffix_len > 0 else ""

                potential_start_idx = -1
                if suffix:
                    for i in range(len(suffix)):
                        sub = suffix[i:]
                        # Check against https and http targets
                        if TARGET_URL_PREFIX.startswith(sub) or TARGET_URL_PREFIX_HTTP.startswith(sub):
                            potential_start_idx = len(text_buffer) - len(suffix) + i
                            break

                if potential_start_idx != -1:
                    # Found partial start, keep it in buffer
                    to_yield += text_buffer[:potential_start_idx]
                    text_buffer = text_buffer[potential_start_idx:]
                else:
                    # No partial start.
                    # Yield based on length or delimiters to ensure smooth streaming
                    # If buffer is long enough (>20 chars), yield it.
                    # Or if it has a space (for English).
                    if len(text_buffer) > 20:
                        to_yield += text_buffer
                        text_buffer = ""
                    elif ' ' in text_buffer:
                         last_space = text_buffer.rfind(' ')
                         to_yield += text_buffer[:last_space+1]
                         text_buffer = text_buffer[last_space+1:]

                if to_yield:
                    parts.append(Part(text=to_yield))

                # Check for images in the chunk
                if hasattr(chunk, "images") and chunk.images:
                    if len(chunk.images) > sent_image_count:
                        # We got new images!
                        # 1. Clear pending URLs as we have better replacements.
                        pending_urls.clear()

                        # 2. Also clear any such URLs currently sitting in text_buffer
                        # Use safe regex to avoid eating Chinese characters
                        url_char_pattern = r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"
                        text_buffer = re.sub(r'https?://googleusercontent\.com' + url_char_pattern + r'*', '', text_buffer)
                        text_buffer = re.sub(r'http://googleusercontent\.com' + url_char_pattern + r'*', '', text_buffer)

                        new_images = chunk.images[sent_image_count:]
                        for img in new_images:
                            try:
                                b64_str, _, _, _, _ = await image_to_base64(img, image_store)
                                parts.append(
                                    Part(
                                        text="", # Explicitly set empty text to ensure client handles it
                                        inline_data=Blob(
                                            mime_type="image/jpeg",
                                            data=b64_str
                                        )
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Failed to process stream image: {e}")
                        sent_image_count = len(chunk.images)

                if parts:
                    if not is_sse and not first:
                        yield ","
                    first = False

                    response_data = GenerateContentResponse(
                        candidates=[
                            Candidate(
                                content=Content(
                                    parts=parts,
                                    role="model"
                                ),
                                finish_reason=None,
                                index=0,
                                safety_ratings=[]
                            )
                        ]
                    )

                    json_str = json.dumps(response_data.model_dump(mode="json", exclude_none=True, by_alias=True))
                    # Add newline after json_str to ensure correct parsing by some clients
                    if is_sse:
                        yield f"data: {json_str}\n\n"
                    else:
                        yield f"{json_str}"

            # End of stream, yield remaining buffer
            # One last check to remove URLs if they were at the very end
            if text_buffer:
                url_char_pattern = r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"
                urls = re.findall(r'https?://googleusercontent\.com' + url_char_pattern + r'*', text_buffer)
                urls += re.findall(r'http://googleusercontent\.com' + url_char_pattern + r'*', text_buffer)

                if urls:
                    pending_urls.extend(urls)
                    text_buffer = re.sub(r'https?://googleusercontent\.com' + url_char_pattern + r'*', '', text_buffer)
                    text_buffer = re.sub(r'http://googleusercontent\.com' + url_char_pattern + r'*', '', text_buffer)

            final_parts = []
            if text_buffer:
                final_parts.append(Part(text=text_buffer))

            # If we still have pending URLs (meaning no images came to clear them), yield them.
            if pending_urls:
                # Use a set to remove duplicates if any
                unique_urls = list(set(pending_urls))
                url_text = " " + " ".join(unique_urls)
                final_parts.append(Part(text=url_text))

            if final_parts:
                if not is_sse and not first:
                     yield ","

                response_data = GenerateContentResponse(
                    candidates=[
                        Candidate(
                            content=Content(parts=final_parts, role="model"),
                            finish_reason=None, index=0, safety_ratings=[]
                        )
                    ]
                )
                json_str = json.dumps(response_data.model_dump(mode="json", exclude_none=True, by_alias=True))
                if is_sse:
                    yield f"data: {json_str}\n\n"
                else:
                    yield f"{json_str}"

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
                yield f"{json_str}"

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
