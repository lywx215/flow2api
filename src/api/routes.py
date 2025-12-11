"""API routes - OpenAI compatible endpoints"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
import base64
import re
import json
import time
from ..core.auth import verify_api_key_header
from ..core.models import ChatCompletionRequest, ImageGenerationRequest, ImageResponse, ImageResponseData
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG, IMAGE_MODELS, map_openai_model_to_internal

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None


def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler


@router.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key_header)):
    """List available models"""
    models = []

    for model_id, config in MODEL_CONFIG.items():
        description = f"{config['type'].capitalize()} generation"
        if config['type'] == 'image':
            description += f" - {config['model_name']}"
        else:
            description += f" - {config['model_key']}"

        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "flow2api",
            "description": description
        })

    return {
        "object": "list",
        "data": models
    }


@router.post("/v1/images/generations")
async def create_image_generation(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key_header)
):
    """Create image from text prompt (OpenAI Images API compatible)

    支持的参数:
    - prompt: 图片描述
    - model: dall-e-3, dall-e-2, gpt-image-1 或内部模型名
    - size: 1024x1024, 1792x1024, 1024x1792
    - response_format: url 或 b64_json
    - n: 生成数量 (目前只支持1)
    """
    try:
        # 将 OpenAI 模型名映射到内部模型
        internal_model = map_openai_model_to_internal(
            request.model or "dall-e-3",
            request.size or "1024x1024"
        )

        # 调用生成方法
        result = await generation_handler.handle_image_generation_simple(
            model=internal_model,
            prompt=request.prompt,
            images=None,  # generations 不使用图片输入
            response_format=request.response_format or "url"
        )

        # 构造 OpenAI 格式响应
        response = ImageResponse(
            created=int(time.time()),
            data=[ImageResponseData(
                url=result.get("url"),
                b64_json=result.get("b64_json"),
                revised_prompt=result.get("revised_prompt")
            )]
        )

        return response.model_dump(exclude_none=True)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/edits")
async def create_image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form("dall-e-2"),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url"),
    mask: Optional[UploadFile] = File(None),
    api_key: str = Depends(verify_api_key_header)
):
    """Edit image based on prompt (OpenAI Images API compatible)

    支持的参数:
    - image: 要编辑的图片 (multipart/form-data)
    - prompt: 编辑描述
    - model: dall-e-2, gpt-image-1 或内部模型名
    - size: 1024x1024, 1792x1024, 1024x1792
    - response_format: url 或 b64_json
    - mask: 可选的蒙版图片
    """
    try:
        # 读取上传的图片
        image_bytes = await image.read()
        images = [image_bytes]

        # 如果有蒙版，也读取它（目前不使用，保留兼容性）
        if mask:
            await mask.read()

        # 将 OpenAI 模型名映射到内部模型
        internal_model = map_openai_model_to_internal(model, size)

        # 调用生成方法
        result = await generation_handler.handle_image_generation_simple(
            model=internal_model,
            prompt=prompt,
            images=images,
            response_format=response_format
        )

        # 构造 OpenAI 格式响应
        response = ImageResponse(
            created=int(time.time()),
            data=[ImageResponseData(
                url=result.get("url"),
                b64_json=result.get("b64_json"),
                revised_prompt=result.get("revised_prompt")
            )]
        )

        return response.model_dump(exclude_none=True)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header)
):
    """Create chat completion (unified endpoint for image and video generation)"""
    try:
        # Extract prompt from messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        last_message = request.messages[-1]
        content = last_message.content

        # Handle both string and array format (OpenAI multimodal)
        prompt = ""
        images: List[bytes] = []

        if isinstance(content, str):
            # Simple text format
            prompt = content
        elif isinstance(content, list):
            # Multimodal format
            for item in content:
                if item.get("type") == "text":
                    prompt = item.get("text", "")
                elif item.get("type") == "image_url":
                    # Extract base64 image
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        # Parse base64
                        match = re.search(r"base64,(.+)", image_url)
                        if match:
                            image_base64 = match.group(1)
                            image_bytes = base64.b64decode(image_base64)
                            images.append(image_bytes)

        # Fallback to deprecated image parameter
        if request.image and not images:
            if request.image.startswith("data:image"):
                match = re.search(r"base64,(.+)", request.image)
                if match:
                    image_base64 = match.group(1)
                    image_bytes = base64.b64decode(image_base64)
                    images.append(image_bytes)

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Call generation handler
        if request.stream:
            # Streaming response
            async def generate():
                async for chunk in generation_handler.handle_generation(
                    model=request.model,
                    prompt=prompt,
                    images=images if images else None,
                    stream=True
                ):
                    yield chunk

                # Send [DONE] signal
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            result = None
            async for chunk in generation_handler.handle_generation(
                model=request.model,
                prompt=prompt,
                images=images if images else None,
                stream=False
            ):
                result = chunk

            if result:
                # Parse the result JSON string
                try:
                    result_json = json.loads(result)
                    return JSONResponse(content=result_json)
                except json.JSONDecodeError:
                    # If not JSON, return as-is
                    return JSONResponse(content={"result": result})
            else:
                raise HTTPException(status_code=500, detail="Generation failed: No response from handler")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

