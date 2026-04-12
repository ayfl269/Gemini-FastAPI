import base64
import hashlib
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from gemini_webapi import ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from loguru import logger

from app.models import (
    GeminiCandidate,
    GeminiContent,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiGenerationConfig,
    GeminiListModelsResponse,
    GeminiModelInfo,
    GeminiPart,
    GeminiTool,
    GeminiUsageMetadata,
    Message,
)
from app.server.middleware import (
    get_image_store_dir,
    get_image_token,
    get_temp_dir,
    verify_gemini_api_key,
)
from app.services import GeminiClientPool, GeminiClientWrapper
from app.utils import g_config
from app.utils.helper import estimate_tokens

router = APIRouter()


def _convert_gemini_content_to_messages(contents: list[GeminiContent]) -> list[Message]:
    """Convert Gemini format contents to internal Message format."""
    messages: list[Message] = []

    for content in contents:
        role = content.role
        if role == "model":
            role = "assistant"

        text_parts: list[str] = []
        tool_calls_list: list[dict[str, Any]] = []
        tool_name: str | None = None
        content_items: list[Any] = []

        for part in content.parts:
            if part.text is not None:
                text_parts.append(part.text)

            if part.inline_data:
                mime_type = part.inline_data.get("mimeType", "image/jpeg")
                b64_data = part.inline_data.get("data", "")
                data_url = f"data:{mime_type};base64,{b64_data}"
                content_items.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )

            if part.file_data:
                file_mime = part.file_data.get("mimeType", "")
                file_b64 = part.file_data.get("fileData", "") if isinstance(part.file_data.get("fileData"), str) else ""
                file_data_url = f"data:{file_mime};base64,{file_b64}"
                content_items.append({"type": "file", "file": {"file_data": file_data_url}})

            if part.function_call:
                func_name = part.function_call.get("name", "")
                func_args_str = part.function_call.get("args", {})
                if isinstance(func_args_str, dict):
                    func_args_str = orjson.dumps(func_args_str).decode("utf-8")

                tool_calls_list.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": func_name, "arguments": func_args_str},
                    }
                )

            if part.function_response:
                tool_name = part.function_response.get("name", "unknown")
                response_data = part.function_response.get("response", {})
                if isinstance(response_data, dict):
                    text_parts.append(orjson.dumps(response_data).decode("utf-8"))
                elif isinstance(response_data, str):
                    text_parts.append(response_data)

        if text_parts and content_items:
            for t in text_parts:
                content_items.insert(0, {"type": "text", "text": t})
            msg = Message(role=role, content=content_items)
        elif content_items:
            msg = Message(role=role, content=content_items)
        else:
            combined_text = "\n".join(text_parts) if text_parts else None
            msg = Message(role=role, content=combined_text)

        if tool_calls_list and role == "assistant":
            from app.models import ToolCall

            msg.tool_calls = [ToolCall.model_validate(tc) for tc in tool_calls_list]

        if role == "tool" and tool_name:
            msg.name = tool_name

        messages.append(msg)

    return messages


def _convert_messages_to_gemini_content(
    messages: list[Message], assistant_text: str | None, tool_calls: list[Any] | None
) -> GeminiContent:
    """Convert internal message format back to Gemini Content format."""
    parts: list[GeminiPart] = []

    if assistant_text:
        parts.append(GeminiPart(text=assistant_text))

    if tool_calls:
        for call in tool_calls:
            func_dict = call.function if hasattr(call, "function") else call.get("function", {})
            func_dict = func_dict.model_dump() if hasattr(func_dict, "model_dump") else dict(func_dict)
            func_args = func_dict.get("arguments", "{}")
            try:
                parsed_args = orjson.loads(func_args) if isinstance(func_args, str) else func_args
            except Exception:
                parsed_args = {}

            parts.append(
                GeminiPart(
                    function_call={
                        "name": func_dict.get("name", ""),
                        "args": parsed_args,
                    }
                )
            )

    return GeminiContent(role="model", parts=parts)


async def _process_response_images(
    response_output, image_store: Path
) -> tuple[list[GeminiPart], str]:
    """
    Process images from model output and convert to Google's official inlineData format.

    Returns:
        tuple: (image_parts_list, image_markdown_for_fallback)

    Google Official Format:
        {
            "inlineData": {
                "mimeType": "image/png",
                "data": "<base64>"
            }
        }
    """
    if not response_output:
        return [], ""

    images = getattr(response_output, 'images', None) or []
    if not images:
        return [], ""
    image_parts: list[GeminiPart] = []
    image_markdown = ""
    seen_hashes = set()

    for image in images:
        try:
            saved_path = await image.save(path=str(image_store), full_size=True) if hasattr(image, 'save') else None
            if not saved_path:
                continue

            original_path = Path(saved_path)
            img_data = original_path.read_bytes()

            suffix = original_path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }.get(suffix, 'image/png')

            b64_data = base64.b64encode(img_data).decode('ascii')
            fhash = hashlib.sha256(img_data).hexdigest()

            if fhash in seen_hashes:
                continue

            seen_hashes.add(fhash)
            fname = original_path.name

            image_parts.append(
                GeminiPart(
                    inline_data={
                        "mimeType": mime_type,
                        "data": b64_data,
                    }
                )
            )

            img_url = f"{image_store.parent.name}/images/{fname}?token={get_image_token(fname)}"
            image_markdown += f"\n\n![{fname}]({img_url})"

        except Exception as exc:
            logger.warning(f"Failed to process output image: {exc}")

    return image_parts, image_markdown


def _extract_system_instruction(request: GeminiGenerateContentRequest) -> str | None:
    """Extract system instruction text from request."""
    if not request.system_instruction:
        return None

    if isinstance(request.system_instruction, dict):
        parts = request.system_instruction.get("parts", [])
        if parts and isinstance(parts, list):
            texts = []
            for p in parts:
                if isinstance(p, dict) and p.get("text"):
                    texts.append(p["text"])
            return "\n".join(texts) if texts else None
        return None

    if isinstance(request.system_instruction, GeminiContent):
        texts = [p.text for p in request.system_instruction.parts if p.text]
        return "\n".join(texts) if texts else None

    return None


def _build_gemini_tool_instruction(gemini_tools: list[GeminiTool] | None) -> str | None:
    """Generate system instruction for Gemini tools in PascalCase protocol format."""
    if not gemini_tools:
        return None

    lines: list[str] = [
        "SYSTEM INTERFACE: You have access to the following technical tools. You MUST invoke them when necessary to fulfill the request, strictly adhering to the provided JSON schemas."
    ]

    for tool in gemini_tools:
        if tool.function_declarations:
            for func_decl in tool.function_declarations:
                name = func_decl.name
                description = func_decl.description or "No description provided."
                lines.append(f"Tool `{name}`: {description}")
                if func_decl.parameters:
                    schema_text = orjson.dumps(func_decl.parameters, option=orjson.OPT_SORT_KEYS).decode("utf-8")
                    lines.append("Arguments JSON schema:")
                    lines.append(schema_text)
                else:
                    lines.append("Arguments JSON schema: {}")

    lines.append(
        "When calling a tool, use this exact format:\n"
        "[ToolCalls]\n"
        "[Call:{function_name}]\n"
        "[CallParameter:{param_name}]\n```\n{value}\n```\n[/CallParameter]\n"
        "... (more parameters as needed) ...\n"
        "[/Call]\n"
        "[/ToolCalls]"
    )

    return "\n".join(lines)


async def _process_gemini_request(
    model_name: str,
    contents: list[GeminiContent],
    generation_config: GeminiGenerationConfig | None,
    system_instruction_text: str | None,
    gemini_tools: list[GeminiTool] | None,
    tmp_dir: Path,
    stream: bool = False,
) -> tuple[AsyncGenerator[ModelOutput] | ModelOutput, ChatSession, Any, Any, list[Message]]:
    """Process a Gemini API request using internal client pool."""
    pool = GeminiClientPool()

    try:
        model = Model.from_name(model_name)
    except ValueError:
        custom_models = {m.model_name: m for m in g_config.gemini.models if m.model_name}
        if model_name in custom_models:
            model = Model.from_dict(custom_models[model_name].model_dump())
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model '{model_name}' not found.") from None

    messages = _convert_gemini_content_to_messages(contents)

    if system_instruction_text:
        messages.insert(0, Message(role="system", content=system_instruction_text))

    tool_instruction = _build_gemini_tool_instruction(gemini_tools)
    if tool_instruction:
        if messages and messages[0].role == "system":
            existing = messages[0].content or ""
            messages[0].content = f"{existing}\n\n{tool_instruction}" if existing else tool_instruction
        else:
            messages.insert(0, Message(role="system", content=tool_instruction))

    try:
        client = await pool.acquire()
        session = client.start_chat(model=model)

        model_input, files = await GeminiClientWrapper.process_conversation(messages, tmp_dir)
    except Exception as e:
        logger.exception("Error preparing conversation for Gemini native API")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e

    use_google_temporary_mode = g_config.gemini.chat_mode == "temporary"
    effective_limit = int(g_config.gemini.max_chars_per_request * 0.9) if use_google_temporary_mode else g_config.gemini.max_chars_per_request

    if len(model_input) > effective_limit:
        logger.warning(f"Input too large ({len(model_input)} chars), converting to file attachment")
        import io

        file_obj = io.BytesIO(model_input.encode("utf-8"))
        file_obj.name = "message.txt"
        final_files = list(files) if files else []
        final_files.append(file_obj)
        model_input = (
            "Context is attached in `message.txt`. "
            "Acknowledge it briefly, then treat it as the primary user input for this turn and answer based on it."
        )
        files = final_files

    try:
        if stream:
            output = session.send_message_stream(model_input, files=files if files else None, temporary=use_google_temporary_mode)
        else:
            output = await session.send_message(model_input, files=files if files else None, temporary=use_google_temporary_mode)
    except Exception as e:
        logger.exception(f"Error sending message to Gemini (native API): {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    return output, session, client, model, messages


def _calculate_gemini_usage(messages: list[Message], output_text: str, thoughts: str | None = None) -> GeminiUsageMetadata:
    """Calculate usage metadata in Gemini format."""
    prompt_tokens = sum(estimate_tokens(_get_msg_text(m)) for m in messages)
    completion_tokens = estimate_tokens(output_text or "")
    if thoughts:
        reasoning_tokens = estimate_tokens(thoughts)
        completion_tokens += reasoning_tokens

    return GeminiUsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=prompt_tokens + completion_tokens,
    )


def _get_msg_text(msg: Message) -> str:
    """Extract text from a message."""
    if isinstance(msg.content, str):
        return msg.content or ""
    elif isinstance(msg.content, list):
        texts = []
        for item in msg.content:
            if item.type == "text" and item.text:
                texts.append(item.text)
        return " ".join(texts)
    return ""


@router.get("/v1beta/models", response_model=GeminiListModelsResponse)
async def list_gemini_models(api_key: str = Depends(verify_gemini_api_key)):
    """List available models in Google Gemini API format."""
    now_models = []
    strategy = g_config.gemini.model_strategy

    custom_models = [m for m in g_config.gemini.models if m.model_name]
    for m in custom_models:
        now_models.append(
            GeminiModelInfo(
                name=f"models/{m.model_name}",
                display_name=m.model_name,
                description="Custom model",
                supported_generation_methods=["generateContent", "streamGenerateContent"],
            )
        )

    if strategy != "overwrite":
        custom_ids = {m.model_name for m in custom_models}
        for model in Model:
            m_name = model.model_name
            if not m_name or m_name == "unspecified":
                continue
            if m_name in custom_ids:
                continue

            now_models.append(
                GeminiModelInfo(
                    name=f"models/{m_name}",
                    version=m_name,
                    display_name=m_name,
                    description=f"Gemini Web model: {m_name}",
                    supported_generation_methods=["generateContent", "streamGenerateContent"],
                )
            )

    return GeminiListModelsResponse(models=now_models)


@router.post("/v1beta/models/{model}:generateContent")
async def generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
    """
    Generate content using the Gemini native API format.
    Endpoint: POST /v1beta/models/{model}:generateContent
    """

    if not request.contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="contents field is required and must not be empty.")

    system_instruction_text = _extract_system_instruction(request)

    try:
        resp_or_stream, _session, _client, _, messages = await _process_gemini_request(
            model_name=model,
            contents=request.contents,
            generation_config=request.generation_config,
            system_instruction_text=system_instruction_text,
            gemini_tools=request.tools,
            tmp_dir=tmp_dir,
            stream=False,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gemini generateContent error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e

    try:
        thoughts = resp_or_stream.thoughts
        raw_clean = GeminiClientWrapper.extract_output(resp_or_stream, include_thoughts=False)
    except Exception as exc:
        logger.exception("Gemini output parsing failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Malformed response."
        ) from exc

    visible_output = raw_clean.strip() if raw_clean else ""

    image_parts, image_markdown = await _process_response_images(resp_or_stream, image_store)

    if image_markdown:
        visible_output += image_markdown

    tool_calls = None
    finish_reason = "STOP"

    if "[ToolCalls]" in visible_output or "[Call:" in visible_output:
        from app.utils.helper import extract_tool_calls as _extract_tc

        _, detected_tool_calls = _extract_tc(visible_output)
        if detected_tool_calls:
            tool_calls = detected_tool_calls
            finish_reason = "STOP"

    all_parts: list[GeminiPart] = []

    if visible_output:
        all_parts.append(GeminiPart(text=visible_output))

    if image_parts:
        all_parts.extend(image_parts)

    if tool_calls:
        for call in tool_calls:
            func_dict = call.function if hasattr(call, "function") else call.get("function", {})
            func_dict = func_dict.model_dump() if hasattr(func_dict, "model_dump") else dict(func_dict)
            func_args = func_dict.get("arguments", "{}")
            try:
                parsed_args = orjson.loads(func_args) if isinstance(func_args, str) else func_args
            except Exception:
                parsed_args = {}
            all_parts.append(
                GeminiPart(
                    function_call={
                        "name": func_dict.get("name", ""),
                        "args": parsed_args,
                    }
                )
            )

    candidate_content = GeminiContent(role="model", parts=all_parts if all_parts else [GeminiPart(text="")])

    usage_metadata = _calculate_gemini_usage(messages, visible_output, thoughts)

    response = GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidate(
                content=candidate_content,
                finish_reason=finish_reason,
                index=0,
            )
        ],
        usage_metadata=usage_metadata,
    )

    return response


@router.post("/v1beta/models/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    request: GeminiGenerateContentRequest,
    raw_request: Request,
    api_key: str = Depends(verify_gemini_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
    image_store: Path = Depends(get_image_store_dir),
):
    """
    Stream generated content using the Gemini native API format.
    Endpoint: POST /v1beta/models/{model}:streamGenerateContent?alt=sse
    Returns Server-Sent Events (SSE) stream compatible with Google's official SDK.
    """
    alt_param = raw_request.query_params.get("alt", "sse").lower()

    if not request.contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="contents field is required and must not be empty.")

    system_instruction_text = _extract_system_instruction(request)

    try:
        generator, _session, _client, _gemini_model, messages = await _process_gemini_request(
            model_name=model,
            contents=request.contents,
            generation_config=request.generation_config,
            system_instruction_text=system_instruction_text,
            gemini_tools=request.tools,
            tmp_dir=tmp_dir,
            stream=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gemini streamGenerateContent error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e

    async def generate_sse_stream():
        full_text = ""
        full_thoughts = ""
        all_outputs: list[ModelOutput] = []

        try:
            async for chunk in generator:
                all_outputs.append(chunk)

                if chunk.thoughts_delta:
                    full_thoughts += chunk.thoughts_delta

                if chunk.text_delta:
                    full_text += chunk.text_delta

                    candidate = GeminiCandidate(
                        content=GeminiContent(role="model", parts=[GeminiPart(text=chunk.text_delta)]),
                        finish_reason=None,
                        index=0,
                    )

                    response_chunk = GeminiGenerateContentResponse(
                        candidates=[candidate],
                    )

                    yield f"data: {orjson.dumps(response_chunk.model_dump(mode='json', by_alias=True)).decode('utf-8')}\n\n"

        except Exception as e:
            logger.exception("Error during Gemini streaming")
            error_response = {
                "error": {
                    "code": 500,
                    "message": str(e),
                    "status": "INTERNAL",
                }
            }
            yield f"data: {orjson.dumps(error_response).decode('utf-8')}\n\n"
            return

        if all_outputs:
            last = all_outputs[-1]
            if last.text:
                full_text = last.text
            if last.thoughts:
                full_thoughts = last.thoughts

        image_parts, image_markdown = await _process_response_images(last if all_outputs else None, image_store)

        if image_markdown:
            full_text += image_markdown

        if image_parts:
            for img_part in image_parts:
                candidate = GeminiCandidate(
                    content=GeminiContent(role="model", parts=[img_part]),
                    finish_reason=None,
                    index=0,
                )
                response_chunk = GeminiGenerateContentResponse(candidates=[candidate])
                yield f"data: {orjson.dumps(response_chunk.model_dump(mode='json', by_alias=True)).decode('utf-8')}\n\n"

        from app.utils.helper import extract_tool_calls as _extract_tc

        _, detected_tool_calls = _extract_tc(full_text)

        if detected_tool_calls:
            for call in detected_tool_calls:
                func_dict = call.function if hasattr(call, "function") else call.get("function", {})
                func_args = func_dict.get("arguments", "{}")
                try:
                    parsed_args = orjson.loads(func_args) if isinstance(func_args, str) else func_args
                except Exception:
                    parsed_args = {}

                tool_candidate = GeminiCandidate(
                    content=GeminiContent(
                        role="model",
                        parts=[
                            GeminiPart(
                                function_call={
                                    "name": func_dict.get("name", ""),
                                    "args": parsed_args,
                                }
                            )
                        ]
                    ),
                    finish_reason=None,
                    index=0,
                )
                tool_chunk = GeminiGenerateContentResponse(candidates=[tool_candidate])
                yield f"data: {orjson.dumps(tool_chunk.model_dump(mode='json', by_alias=True)).decode('utf-8')}\n\n"

            finish_reason = "STOP"
        else:
            finish_reason = "STOP"

        usage_metadata = _calculate_gemini_usage(messages, full_text, full_thoughts)

        final_candidate = GeminiCandidate(
            content=GeminiContent(role="model", parts=[]),
            finish_reason=finish_reason,
            index=0,
        )

        final_response = GeminiGenerateContentResponse(
            candidates=[final_candidate],
            usage_metadata=usage_metadata,
        )

        yield f"data: {orjson.dumps(final_response.model_dump(mode='json', by_alias=True)).decode('utf-8')}\n\n"
        yield "data: {}\n\n"

    media_type = "text/event-stream" if alt_param == "sse" else "application/json"
    return StreamingResponse(generate_sse_stream(), media_type=media_type)
