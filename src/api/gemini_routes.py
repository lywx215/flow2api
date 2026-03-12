"""Gemini native format API routes for Flow2API.

Provides endpoints compatible with the Gemini API format:
- POST /v1beta/models/{model}:generateContent
- GET  /v1beta/models
"""
import asyncio
import base64
import json
import re
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ..core.auth import verify_api_key_header
from ..core.logger import debug_logger
from ..services.generation_handler import (
    GenerationHandler, MODEL_CONFIG, resolve_compact_model,
    COMPACT_IMAGE_MODELS, COMPACT_VIDEO_MODELS,
    RESPONSE_FORMAT_B64, RESPONSE_FORMAT_URL,
)

router = APIRouter()

# Will be set by main.py
generation_handler: GenerationHandler = None


def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler


def _parse_gemini_contents(contents: List[Dict]) -> tuple:
    """Parse Gemini native 'contents' field into (prompt, images).

    Returns:
        (prompt: str, images: list[bytes])
    """
    prompt = ""
    images: List[bytes] = []

    for content_item in contents:
        parts = content_item.get("parts", [])
        for part in parts:
            if "text" in part:
                prompt = part["text"]
            elif "inlineData" in part:
                inline = part["inlineData"]
                data_b64 = inline.get("data", "")
                if data_b64:
                    images.append(base64.b64decode(data_b64))

    return prompt, images


def _parse_generation_config(gen_config: Optional[Dict]) -> Dict[str, Any]:
    """Parse Gemini generationConfig into internal parameters.

    Returns dict with keys: aspect_ratio, resolution, response_format
    """
    result = {
        "aspect_ratio": None,
        "resolution": None,
        "response_format": RESPONSE_FORMAT_B64,
    }

    if not gen_config:
        return result

    # 图片配置
    image_config = gen_config.get("imageConfig") or {}
    if image_config:
        result["aspect_ratio"] = image_config.get("aspectRatio")
        image_size = image_config.get("imageSize")
        if image_size:
            result["resolution"] = image_size
        output_format = image_config.get("outputFormat")
        if output_format:
            result["response_format"] = output_format

    # 视频配置
    video_config = gen_config.get("videoConfig") or {}
    if video_config:
        result["aspect_ratio"] = video_config.get("aspectRatio")
        video_res = video_config.get("resolution")
        if video_res:
            result["resolution"] = video_res

    return result


def _build_gemini_image_response(
    image_data: bytes,
    mime_type: str,
    text: str,
    response_format: str,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Gemini native format response for image generation."""
    parts = []
    if text:
        parts.append({"text": text})

    if response_format == RESPONSE_FORMAT_URL and image_url:
        parts.append({
            "fileData": {
                "fileUri": image_url,
                "mimeType": mime_type,
            }
        })
    else:
        # Default: inlineData (b64_json)
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": base64.b64encode(image_data).decode("utf-8") if isinstance(image_data, bytes) else image_data,
            }
        })

    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        }
    }


def _build_gemini_video_response(
    video_url: str,
    text: str,
) -> Dict[str, Any]:
    """Build a Gemini native format response for video generation."""
    parts = []
    if text:
        parts.append({"text": text})

    parts.append({
        "fileData": {
            "fileUri": video_url,
            "mimeType": "video/mp4",
        }
    })

    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        }
    }


def _build_gemini_error_response(error_msg: str) -> Dict[str, Any]:
    """Build a Gemini native format error response."""
    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": f"Error: {error_msg}"}],
            },
            "finishReason": "ERROR",
        }],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        }
    }


@router.get("/v1beta/models")
async def gemini_list_models(api_key: str = Depends(verify_api_key_header)):
    """List available models in Gemini format."""
    models = []

    # 集约图片模型
    for model_id, model_name in COMPACT_IMAGE_MODELS.items():
        models.append({
            "name": f"models/{model_id}",
            "displayName": model_id,
            "description": f"Image generation - {model_name}",
            "supportedGenerationMethods": ["generateContent"],
        })

    # 集约视频模型
    for model_id, video_def in COMPACT_VIDEO_MODELS.items():
        models.append({
            "name": f"models/{model_id}",
            "displayName": model_id,
            "description": f"Video generation - {video_def['video_type']}",
            "supportedGenerationMethods": ["generateContent"],
        })

    # 旧模型
    for model_id, config in MODEL_CONFIG.items():
        desc = f"{config['type'].capitalize()} generation"
        if config['type'] == 'image':
            desc += f" - {config['model_name']}"
        else:
            desc += f" - {config['model_key']}"
        models.append({
            "name": f"models/{model_id}",
            "displayName": model_id,
            "description": desc,
            "supportedGenerationMethods": ["generateContent"],
        })

    return {"models": models}


@router.post("/v1beta/models/{model_action:path}")
async def gemini_generate_content(
    model_action: str,
    request: Request,
    api_key: str = Depends(verify_api_key_header),
):
    """Gemini native format: POST /v1beta/models/{model}:generateContent

    Supports both image and video generation.
    """
    # Parse model name from path (e.g. "gemini-3.1-flash-image:generateContent")
    if ":generateContent" not in model_action:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {model_action}")

    model = model_action.replace(":generateContent", "").strip("/")
    if not model:
        raise HTTPException(status_code=400, detail="Model name is required")

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    contents = body.get("contents")
    if not contents:
        raise HTTPException(status_code=400, detail="'contents' field is required")

    gen_config = body.get("generationConfig")

    # Parse contents → prompt + images
    prompt, images = _parse_gemini_contents(contents)
    if not prompt:
        raise HTTPException(status_code=400, detail="No text prompt found in contents")

    # Parse generation config → aspect_ratio, resolution, response_format
    params = _parse_generation_config(gen_config)
    aspect_ratio = params["aspect_ratio"]
    resolution = params["resolution"]
    response_format = params["response_format"]

    # Resolve model
    resolved_key, model_config = resolve_compact_model(model, aspect_ratio, resolution)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    debug_logger.log_info(
        f"[GEMINI] generateContent - model: {model}, resolved: {resolved_key}, "
        f"aspect_ratio: {aspect_ratio}, resolution: {resolution}, format: {response_format}"
    )

    # Call generation handler (must use stream=True — non-stream only checks availability)
    collected_text = []
    error_msg = None

    try:
        async for chunk in generation_handler.handle_generation(
            model=model,
            prompt=prompt,
            images=images if images else None,
            stream=True,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        ):
            # Parse SSE chunks and accumulate text
            if isinstance(chunk, str):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            for choice in choices:
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    collected_text.append(content)
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        error_msg = str(e)
        debug_logger.log_error(f"[GEMINI] Generation error: {error_msg}")

    if error_msg:
        return JSONResponse(
            content=_build_gemini_error_response(error_msg),
            status_code=500,
        )

    # Join ALL collected text, then parse URLs from the COMPLETE text
    full_text = "".join(collected_text)
    collected_image_url = None
    collected_video_url = None

    debug_logger.log_info(
        f"[GEMINI] Collected text length: {len(full_text)}, chunks: {len(collected_text)}"
    )

    # Parse image/video URLs from the full text (now complete, no chunk splitting)
    if full_text:
        # Image URL from markdown: ![...](https://...) or ![...](data:image/...;base64,...)
        img_match = re.search(r"!\[.*?\]\(((https?://|data:image/)[^\)]+)\)", full_text)
        if img_match:
            collected_image_url = img_match.group(1)
        else:
            debug_logger.log_warning(
                f"[GEMINI] No image URL found in text. First 200 chars: {full_text[:200]}"
            )

        # Video URL from markdown or plain URL
        vid_match = re.search(r"\[.*?(?:视频|video|下载).*?\]\((https?://[^\)]+)\)", full_text, re.IGNORECASE)
        if vid_match:
            collected_video_url = vid_match.group(1)
        else:
            vid_ext_match = re.search(r"(https?://[^\s\)]+\.mp4[^\s\)]*)", full_text)
            if vid_ext_match:
                collected_video_url = vid_ext_match.group(1)

    gen_type = model_config.get("type", "image")

    # Build response based on type
    if gen_type == "image" and collected_image_url:
        # Clean text: remove markdown image syntax (both http and data: URIs)
        clean_text = re.sub(r"!\[.*?\]\((https?://|data:image/)[^\)]+\)", "", full_text).strip()

        # Check if image is a data URI (base64 inline)
        if collected_image_url.startswith("data:image/"):
            # Extract MIME type and base64 data from data URI
            data_uri_match = re.match(r"data:(image/[^;]+);base64,(.*)", collected_image_url)
            if data_uri_match:
                mime = data_uri_match.group(1)
                b64_data = data_uri_match.group(2)

                if response_format == RESPONSE_FORMAT_URL:
                    # User wants URL but we only have data URI — return as inlineData anyway
                    # (no HTTP URL available for data URI images)
                    debug_logger.log_info("[GEMINI] Image is data URI, cannot return as URL, returning inlineData")

                return JSONResponse(content={
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [
                                *([{"text": clean_text}] if clean_text else []),
                                {"inlineData": {"mimeType": mime, "data": b64_data}},
                            ],
                        },
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 0,
                        "candidatesTokenCount": 0,
                        "totalTokenCount": 0,
                    }
                })

        # HTTP URL image
        if response_format == RESPONSE_FORMAT_URL:
            return JSONResponse(content=_build_gemini_image_response(
                image_data=b"",
                mime_type="image/png",
                text=clean_text,
                response_format=RESPONSE_FORMAT_URL,
                image_url=collected_image_url,
            ))
        else:
            # Download image and return base64
            try:
                from ..api.routes import retrieve_image_data
                image_bytes = await retrieve_image_data(collected_image_url)
                if image_bytes:
                    # Detect MIME type
                    mime = "image/png"
                    if image_bytes[:3] == b'\xff\xd8\xff':
                        mime = "image/jpeg"
                    elif image_bytes[:4] == b'\x89PNG':
                        mime = "image/png"
                    elif image_bytes[:4] == b'RIFF':
                        mime = "image/webp"

                    return JSONResponse(content=_build_gemini_image_response(
                        image_data=image_bytes,
                        mime_type=mime,
                        text=clean_text,
                        response_format=RESPONSE_FORMAT_B64,
                    ))
            except Exception as e:
                debug_logger.log_warning(f"[GEMINI] Failed to download image for b64: {e}")

            # Fallback to URL if download fails
            return JSONResponse(content=_build_gemini_image_response(
                image_data=b"",
                mime_type="image/png",
                text=clean_text,
                response_format=RESPONSE_FORMAT_URL,
                image_url=collected_image_url,
            ))

    elif gen_type == "video" and collected_video_url:
        clean_text = re.sub(r"\[.*?\]\(https?://[^\)]+\)", "", full_text).strip()
        return JSONResponse(content=_build_gemini_video_response(
            video_url=collected_video_url,
            text=clean_text,
        ))

    elif full_text:
        # Fallback: return text response (e.g. error message from handler)
        return JSONResponse(content={
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": full_text}],
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0,
            }
        })
    else:
        return JSONResponse(
            content=_build_gemini_error_response("No content generated"),
            status_code=500,
        )

