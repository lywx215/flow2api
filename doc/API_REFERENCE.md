# Flow2API API 文档

> Flow2API 提供两种 API 格式：**OpenAI 兼容格式** 和 **Gemini 原生格式**。
> 所有端点均需要通过 **Bearer Token** 进行认证。

---

## 目录

- [认证](#认证)
- [OpenAI 兼容格式](#openai-兼容格式)
  - [获取模型列表](#1-获取模型列表)
  - [创建聊天补全 (Chat Completions)](#2-创建聊天补全)
- [Gemini 原生格式](#gemini-原生格式)
  - [获取模型列表 (Gemini)](#3-获取模型列表-gemini)
  - [生成内容 (generateContent)](#4-生成内容)
- [集约模型名](#集约模型名)
  - [图片模型](#图片模型集约)
  - [视频模型](#视频模型集约)
- [完整模型名参考表](#完整模型名参考表)
  - [图片模型 (完整)](#图片模型完整)
  - [视频模型 (完整)](#视频模型完整)
- [参数参考](#参数参考)
  - [aspect_ratio](#aspect_ratio-宽高比)
  - [resolution](#resolution-分辨率)
  - [response_format](#response_format-响应格式)
- [错误码](#错误码)

---

## 认证

所有 API 请求均需要在 HTTP Header 中携带 Bearer Token：

```
Authorization: Bearer <your-api-key>
```

API Key 在首次登录管理后台 (`http://localhost:8000`) 后可在系统配置中查看或修改。

---

## OpenAI 兼容格式

### 1. 获取模型列表

获取所有可用的模型列表（包括集约模型名和完整模型名）。

**请求**

```
GET /v1/models
```

**请求头**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| Authorization | string | ✅ | `Bearer <api-key>` |

**响应示例**

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemini-3.1-flash-image",
      "object": "model",
      "owned_by": "flow2api",
      "description": "Image generation (compact) - NARWHAL"
    },
    {
      "id": "veo-3.1-fast",
      "object": "model",
      "owned_by": "flow2api",
      "description": "Video generation (compact) - t2v"
    },
    {
      "id": "gemini-3.1-flash-image-landscape-2k",
      "object": "model",
      "owned_by": "flow2api",
      "description": "Image generation - NARWHAL"
    }
  ]
}
```

---

### 2. 创建聊天补全

统一的图片/视频生成入口，兼容 OpenAI Chat Completions 格式。

**请求**

```
POST /v1/chat/completions
```

**请求头**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| Authorization | string | ✅ | `Bearer <api-key>` |
| Content-Type | string | ✅ | `application/json` |

**请求体参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | ✅ | - | 模型名称，支持集约模型名或完整模型名 |
| `messages` | array | ✅* | - | OpenAI 格式的消息数组，与 `contents` 二选一 |
| `contents` | array | ✅* | - | Gemini 原生格式的内容数组，与 `messages` 二选一 |
| `stream` | boolean | ❌ | `false` | 是否流式输出。**生成功能必须设为 `true`** |
| `aspect_ratio` | string | ❌ | `"16:9"` | 宽高比，仅集约模型名时生效 |
| `resolution` | string | ❌ | `null` | 分辨率/放大倍数，仅集约模型名时生效 |
| `response_format` | string | ❌ | `"b64_json"` | 图片响应格式：`"b64_json"` 或 `"url"` |
| `image` | string | ❌ | `null` | Base64 编码的图片（已废弃，请用 messages） |

> **注意**: `messages` 和 `contents` 至少填写一个。当两者都提供时，优先使用 `messages`。

**messages 格式**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "文本提示词"
    }
  ]
}
```

或多模态格式（包含图片）：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "将这张图片变成水彩画风格"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64编码的图片>"
          }
        }
      ]
    }
  ]
}
```

**contents 格式（Gemini 原生）**

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "一只可爱的猫咪"},
        {"inlineData": {"mimeType": "image/jpeg", "data": "<base64>"}}
      ]
    }
  ]
}
```

**image_url 支持的格式**

| 格式 | 示例 | 说明 |
|------|------|------|
| Base64 Data URI | `data:image/jpeg;base64,/9j/4AAQ...` | 内联 Base64 编码图片 |
| HTTP/HTTPS URL | `https://example.com/image.jpg` | 远程图片 URL（服务端下载） |

---

#### 2.1 文生图

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "aspect_ratio": "16:9",
    "resolution": "2K",
    "messages": [
      {
        "role": "user",
        "content": "一只可爱的猫咪在花园里玩耍"
      }
    ],
    "stream": true
  }'
```

#### 2.2 图生图

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "aspect_ratio": "16:9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "将这张图片变成水彩画风格"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,<base64编码的图片>"
            }
          }
        ]
      }
    ],
    "stream": true
  }'
```

#### 2.3 文生视频

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-fast",
    "aspect_ratio": "16:9",
    "messages": [
      {
        "role": "user",
        "content": "一只小猫在草地上追逐蝴蝶"
      }
    ],
    "stream": true
  }'
```

#### 2.4 图生视频（首尾帧）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-i2v",
    "aspect_ratio": "16:9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "从第一张图过渡到第二张图"
          },
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,<首帧base64>"}
          },
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,<尾帧base64>"}
          }
        ]
      }
    ],
    "stream": true
  }'
```

#### 2.5 多图生成视频（R2V）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-r2v",
    "aspect_ratio": "9:16",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "以参考图为基础生成镜头平滑推进的竖屏视频"
          },
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<参考图1>"}},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<参考图2>"}},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<参考图3>"}}
        ]
      }
    ],
    "stream": true
  }'
```

#### 2.6 视频放大（4K/1080P）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-fast",
    "aspect_ratio": "16:9",
    "resolution": "4k",
    "messages": [
      {
        "role": "user",
        "content": "一只小猫在草地上追逐蝴蝶"
      }
    ],
    "stream": true
  }'
```

#### 2.7 连续对话图片生成

系统会自动从历史 assistant 消息中提取上一次生成的图片作为参考图：

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "messages": [
      {
        "role": "user",
        "content": "画一只猫"
      },
      {
        "role": "assistant",
        "content": "![猫](http://localhost:8000/tmp/abc123.png)"
      },
      {
        "role": "user",
        "content": "给这只猫加上帽子"
      }
    ],
    "stream": true
  }'
```

**流式响应格式 (SSE)**

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"gemini-3.1-flash-image","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"gemini-3.1-flash-image","choices":[{"index":0,"delta":{"content":"✨ 图片生成任务已启动\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1710000000,"model":"gemini-3.1-flash-image","choices":[{"index":0,"delta":{"content":"![image](http://localhost:8000/tmp/result.png)"},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Gemini 原生格式

### 3. 获取模型列表 (Gemini)

**请求**

```
GET /v1beta/models
```

**请求头**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| Authorization | string | ✅ | `Bearer <api-key>` |

**响应示例**

```json
{
  "models": [
    {
      "name": "models/gemini-3.1-flash-image",
      "displayName": "gemini-3.1-flash-image",
      "description": "Image generation - NARWHAL",
      "supportedGenerationMethods": ["generateContent"]
    },
    {
      "name": "models/veo-3.1-fast",
      "displayName": "veo-3.1-fast",
      "description": "Video generation - t2v",
      "supportedGenerationMethods": ["generateContent"]
    }
  ]
}
```

---

### 4. 生成内容

**请求**

```
POST /v1beta/models/{model}:generateContent
```

**路径参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | ✅ | 模型名称（集约模型名或完整模型名） |

**请求头**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| Authorization | string | ✅ | `Bearer <api-key>` |
| Content-Type | string | ✅ | `application/json` |

**请求体参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `contents` | array | ✅ | Gemini 格式的内容数组 |
| `generationConfig` | object | ❌ | 生成配置（宽高比、分辨率、输出格式等） |

**contents 格式**

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "提示词文本"},
        {"inlineData": {"mimeType": "image/jpeg", "data": "<base64编码>"}}
      ]
    }
  ]
}
```

**contents[].parts 支持的类型**

| 类型 | 格式 | 说明 |
|------|------|------|
| 文本 | `{"text": "..."}` | 提示词文本 |
| 内联图片 | `{"inlineData": {"mimeType": "image/jpeg", "data": "..."}}` | Base64 编码图片 |

**generationConfig 参数**

```json
{
  "generationConfig": {
    "responseModalities": ["IMAGE"],
    "imageConfig": {
      "aspectRatio": "16:9",
      "imageSize": "2K",
      "outputFormat": "b64_json"
    },
    "videoConfig": {
      "aspectRatio": "16:9",
      "resolution": "4k"
    }
  }
}
```

| 参数路径 | 类型 | 说明 |
|---------|------|------|
| `responseModalities` | array | 响应类型，可选 `["IMAGE"]` 或 `["VIDEO"]` |
| `imageConfig.aspectRatio` | string | 图片宽高比，见 [aspect_ratio 参数](#aspect_ratio-宽高比) |
| `imageConfig.imageSize` | string | 图片分辨率：`"1K"`, `"2K"`, `"4K"` |
| `imageConfig.outputFormat` | string | 输出格式：`"b64_json"`（默认）或 `"url"` |
| `videoConfig.aspectRatio` | string | 视频宽高比，见 [aspect_ratio 参数](#aspect_ratio-宽高比) |
| `videoConfig.resolution` | string | 视频放大分辨率：`"1080p"` 或 `"4k"` |

---

#### 4.1 Gemini 格式 - 文生图

```bash
curl -X POST "http://localhost:8000/v1beta/models/gemini-3.1-flash-image:generateContent" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "一只可爱的猫咪在花园里玩耍"}]
      }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": {
        "aspectRatio": "16:9",
        "imageSize": "2K",
        "outputFormat": "b64_json"
      }
    }
  }'
```

**响应 (outputFormat=b64_json)**

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {"text": "生成描述文本"},
          {
            "inlineData": {
              "mimeType": "image/png",
              "data": "<base64编码的图片数据>"
            }
          }
        ]
      },
      "finishReason": "STOP"
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 0,
    "candidatesTokenCount": 0,
    "totalTokenCount": 0
  }
}
```

**响应 (outputFormat=url)**

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {"text": "生成描述文本"},
          {
            "fileData": {
              "fileUri": "http://localhost:8000/tmp/result.png",
              "mimeType": "image/png"
            }
          }
        ]
      },
      "finishReason": "STOP"
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 0,
    "candidatesTokenCount": 0,
    "totalTokenCount": 0
  }
}
```

#### 4.2 Gemini 格式 - 图生图

```bash
curl -X POST "http://localhost:8000/v1beta/models/gemini-3.1-flash-image:generateContent" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {"text": "将这张图片变成水彩画风格"},
          {"inlineData": {"mimeType": "image/jpeg", "data": "<base64编码的图片>"}}
        ]
      }
    ],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": {"aspectRatio": "16:9"}
    }
  }'
```

#### 4.3 Gemini 格式 - 文生视频

```bash
curl -X POST "http://localhost:8000/v1beta/models/veo-3.1-fast:generateContent" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "一只小猫在草地上追逐蝴蝶"}]
      }
    ],
    "generationConfig": {
      "responseModalities": ["VIDEO"],
      "videoConfig": {
        "aspectRatio": "16:9"
      }
    }
  }'
```

**视频响应**

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {"text": "视频生成完成"},
          {
            "fileData": {
              "fileUri": "http://localhost:8000/tmp/video_result.mp4",
              "mimeType": "video/mp4"
            }
          }
        ]
      },
      "finishReason": "STOP"
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 0,
    "candidatesTokenCount": 0,
    "totalTokenCount": 0
  }
}
```

#### 4.4 Gemini 格式 - 图生视频

```bash
curl -X POST "http://localhost:8000/v1beta/models/veo-3.1-i2v:generateContent" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {"text": "让画面中的人物走动起来"},
          {"inlineData": {"mimeType": "image/jpeg", "data": "<首帧base64>"}},
          {"inlineData": {"mimeType": "image/jpeg", "data": "<尾帧base64>"}}
        ]
      }
    ],
    "generationConfig": {
      "responseModalities": ["VIDEO"],
      "videoConfig": {"aspectRatio": "16:9"}
    }
  }'
```

#### 4.5 Gemini 格式 - 视频放大

```bash
curl -X POST "http://localhost:8000/v1beta/models/veo-3.1-fast:generateContent" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "城市夜景延时摄影"}]
      }
    ],
    "generationConfig": {
      "responseModalities": ["VIDEO"],
      "videoConfig": {
        "aspectRatio": "16:9",
        "resolution": "4k"
      }
    }
  }'
```

---

## 集约模型名

集约模型名将 70+ 个完整模型名压缩为 19 个，尺寸和分辨率通过参数传递。

### 图片模型（集约）

| 集约模型名 | 内部引擎 | 说明 |
|-----------|---------|------|
| `gemini-2.5-flash-image` | GEM_PIX | Gemini 2.5 Flash 图片生成 |
| `gemini-3.0-pro-image` | GEM_PIX_2 | Gemini 3.0 Pro 图片生成 |
| `gemini-3.1-flash-image` | NARWHAL | Gemini 3.1 Flash 图片生成（最新） |
| `imagen-4.0-generate-preview` | IMAGEN_3_5 | Imagen 4.0 图片生成 |

使用时搭配 `aspect_ratio` 和 `resolution` 参数：

```json
{
  "model": "gemini-3.1-flash-image",
  "aspect_ratio": "16:9",
  "resolution": "2K"
}
```

### 视频模型（集约）

#### 文生视频 (T2V)

| 集约模型名 | 说明 |
|-----------|------|
| `veo-3.1-fast` | Veo 3.1 快速文生视频 |
| `veo-3.1-fast-ultra` | Veo 3.1 Ultra 品质 |
| `veo-3.1-fast-ultra-relaxed` | Veo 3.1 Ultra 放松模式 |
| `veo-3.1` | Veo 3.1 标准 |
| `veo-2.1` | Veo 2.1 |
| `veo-2.0` | Veo 2.0 |

#### 图生视频 (I2V) — 支持 1-2 张图片

| 集约模型名 | 说明 |
|-----------|------|
| `veo-3.1-i2v` | Veo 3.1 首尾帧视频 |
| `veo-3.1-i2v-ultra` | Veo 3.1 首尾帧 Ultra 品质 |
| `veo-3.1-i2v-ultra-relaxed` | Veo 3.1 首尾帧放松模式 |
| `veo-3.1-i2v-standard` | Veo 3.1 首尾帧标准 |
| `veo-2.1-i2v` | Veo 2.1 首尾帧 |
| `veo-2.0-i2v` | Veo 2.0 首尾帧 |

#### 多图生成视频 (R2V) — 支持最多 3 张参考图

| 集约模型名 | 说明 |
|-----------|------|
| `veo-3.1-r2v` | Veo 3.1 多图视频 |
| `veo-3.1-r2v-ultra` | Veo 3.1 多图 Ultra 品质 |
| `veo-3.1-r2v-ultra-relaxed` | Veo 3.1 多图放松模式 |

---

## 完整模型名参考表

> 使用完整模型名时，无需传递 `aspect_ratio` / `resolution` 参数，因为尺寸和分辨率已包含在模型名中。

### 图片模型（完整）

| 模型名称 | 引擎 | 尺寸 | 分辨率 |
|---------|------|------|--------|
| `gemini-2.5-flash-image-landscape` | GEM_PIX | 横屏 16:9 | 1K |
| `gemini-2.5-flash-image-portrait` | GEM_PIX | 竖屏 9:16 | 1K |
| `gemini-3.0-pro-image-landscape` | GEM_PIX_2 | 横屏 16:9 | 1K |
| `gemini-3.0-pro-image-portrait` | GEM_PIX_2 | 竖屏 9:16 | 1K |
| `gemini-3.0-pro-image-square` | GEM_PIX_2 | 方图 1:1 | 1K |
| `gemini-3.0-pro-image-four-three` | GEM_PIX_2 | 横屏 4:3 | 1K |
| `gemini-3.0-pro-image-three-four` | GEM_PIX_2 | 竖屏 3:4 | 1K |
| `gemini-3.0-pro-image-landscape-2k` | GEM_PIX_2 | 横屏 16:9 | 2K |
| `gemini-3.0-pro-image-portrait-2k` | GEM_PIX_2 | 竖屏 9:16 | 2K |
| `gemini-3.0-pro-image-square-2k` | GEM_PIX_2 | 方图 1:1 | 2K |
| `gemini-3.0-pro-image-four-three-2k` | GEM_PIX_2 | 横屏 4:3 | 2K |
| `gemini-3.0-pro-image-three-four-2k` | GEM_PIX_2 | 竖屏 3:4 | 2K |
| `gemini-3.0-pro-image-landscape-4k` | GEM_PIX_2 | 横屏 16:9 | 4K |
| `gemini-3.0-pro-image-portrait-4k` | GEM_PIX_2 | 竖屏 9:16 | 4K |
| `gemini-3.0-pro-image-square-4k` | GEM_PIX_2 | 方图 1:1 | 4K |
| `gemini-3.0-pro-image-four-three-4k` | GEM_PIX_2 | 横屏 4:3 | 4K |
| `gemini-3.0-pro-image-three-four-4k` | GEM_PIX_2 | 竖屏 3:4 | 4K |
| `imagen-4.0-generate-preview-landscape` | IMAGEN_3_5 | 横屏 16:9 | 1K |
| `imagen-4.0-generate-preview-portrait` | IMAGEN_3_5 | 竖屏 9:16 | 1K |
| `gemini-3.1-flash-image-landscape` | NARWHAL | 横屏 16:9 | 1K |
| `gemini-3.1-flash-image-portrait` | NARWHAL | 竖屏 9:16 | 1K |
| `gemini-3.1-flash-image-square` | NARWHAL | 方图 1:1 | 1K |
| `gemini-3.1-flash-image-four-three` | NARWHAL | 横屏 4:3 | 1K |
| `gemini-3.1-flash-image-three-four` | NARWHAL | 竖屏 3:4 | 1K |
| `gemini-3.1-flash-image-landscape-2k` | NARWHAL | 横屏 16:9 | 2K |
| `gemini-3.1-flash-image-portrait-2k` | NARWHAL | 竖屏 9:16 | 2K |
| `gemini-3.1-flash-image-square-2k` | NARWHAL | 方图 1:1 | 2K |
| `gemini-3.1-flash-image-four-three-2k` | NARWHAL | 横屏 4:3 | 2K |
| `gemini-3.1-flash-image-three-four-2k` | NARWHAL | 竖屏 3:4 | 2K |
| `gemini-3.1-flash-image-landscape-4k` | NARWHAL | 横屏 16:9 | 4K |
| `gemini-3.1-flash-image-portrait-4k` | NARWHAL | 竖屏 9:16 | 4K |
| `gemini-3.1-flash-image-square-4k` | NARWHAL | 方图 1:1 | 4K |
| `gemini-3.1-flash-image-four-three-4k` | NARWHAL | 横屏 4:3 | 4K |
| `gemini-3.1-flash-image-three-four-4k` | NARWHAL | 竖屏 3:4 | 4K |

### 视频模型（完整）

#### 文生视频 (T2V) — 不支持上传图片

| 模型名称 | 尺寸 |
|---------|------|
| `veo_3_1_t2v_fast_portrait` | 竖屏 |
| `veo_3_1_t2v_fast_landscape` | 横屏 |
| `veo_2_1_fast_d_15_t2v_portrait` | 竖屏 |
| `veo_2_1_fast_d_15_t2v_landscape` | 横屏 |
| `veo_2_0_t2v_portrait` | 竖屏 |
| `veo_2_0_t2v_landscape` | 横屏 |
| `veo_3_1_t2v_fast_portrait_ultra` | 竖屏 Ultra |
| `veo_3_1_t2v_fast_ultra` | 横屏 Ultra |
| `veo_3_1_t2v_fast_portrait_ultra_relaxed` | 竖屏 Ultra Relaxed |
| `veo_3_1_t2v_fast_ultra_relaxed` | 横屏 Ultra Relaxed |
| `veo_3_1_t2v_portrait` | 竖屏 标准 |
| `veo_3_1_t2v_landscape` | 横屏 标准 |

#### 首尾帧 (I2V) — 支持 1-2 张图片

| 模型名称 | 尺寸 |
|---------|------|
| `veo_3_1_i2v_s_fast_portrait_fl` | 竖屏 |
| `veo_3_1_i2v_s_fast_fl` | 横屏 |
| `veo_2_1_fast_d_15_i2v_portrait` | 竖屏 |
| `veo_2_1_fast_d_15_i2v_landscape` | 横屏 |
| `veo_2_0_i2v_portrait` | 竖屏 |
| `veo_2_0_i2v_landscape` | 横屏 |
| `veo_3_1_i2v_s_fast_portrait_ultra_fl` | 竖屏 Ultra |
| `veo_3_1_i2v_s_fast_ultra_fl` | 横屏 Ultra |
| `veo_3_1_i2v_s_fast_portrait_ultra_relaxed` | 竖屏 Ultra Relaxed |
| `veo_3_1_i2v_s_fast_ultra_relaxed` | 横屏 Ultra Relaxed |
| `veo_3_1_i2v_s_portrait` | 竖屏 标准 |
| `veo_3_1_i2v_s_landscape` | 横屏 标准 |

#### 多图生成 (R2V) — 支持最多 3 张参考图

| 模型名称 | 尺寸 |
|---------|------|
| `veo_3_1_r2v_fast_portrait` | 竖屏 |
| `veo_3_1_r2v_fast` | 横屏 |
| `veo_3_1_r2v_fast_portrait_ultra` | 竖屏 Ultra |
| `veo_3_1_r2v_fast_ultra` | 横屏 Ultra |
| `veo_3_1_r2v_fast_portrait_ultra_relaxed` | 竖屏 Ultra Relaxed |
| `veo_3_1_r2v_fast_ultra_relaxed` | 横屏 Ultra Relaxed |

#### 视频放大 (Upsample)

| 模型名称 | 输出 |
|---------|------|
| `veo_3_1_t2v_fast_portrait_4k` | T2V 竖屏 4K |
| `veo_3_1_t2v_fast_4k` | T2V 横屏 4K |
| `veo_3_1_t2v_fast_portrait_ultra_4k` | T2V 竖屏 Ultra 4K |
| `veo_3_1_t2v_fast_ultra_4k` | T2V 横屏 Ultra 4K |
| `veo_3_1_t2v_fast_portrait_1080p` | T2V 竖屏 1080P |
| `veo_3_1_t2v_fast_1080p` | T2V 横屏 1080P |
| `veo_3_1_t2v_fast_portrait_ultra_1080p` | T2V 竖屏 Ultra 1080P |
| `veo_3_1_t2v_fast_ultra_1080p` | T2V 横屏 Ultra 1080P |
| `veo_3_1_i2v_s_fast_portrait_ultra_fl_4k` | I2V 竖屏 Ultra 4K |
| `veo_3_1_i2v_s_fast_ultra_fl_4k` | I2V 横屏 Ultra 4K |
| `veo_3_1_i2v_s_fast_portrait_ultra_fl_1080p` | I2V 竖屏 Ultra 1080P |
| `veo_3_1_i2v_s_fast_ultra_fl_1080p` | I2V 横屏 Ultra 1080P |
| `veo_3_1_r2v_fast_portrait_ultra_4k` | R2V 竖屏 Ultra 4K |
| `veo_3_1_r2v_fast_ultra_4k` | R2V 横屏 Ultra 4K |
| `veo_3_1_r2v_fast_portrait_ultra_1080p` | R2V 竖屏 Ultra 1080P |
| `veo_3_1_r2v_fast_ultra_1080p` | R2V 横屏 Ultra 1080P |

---

## 参数参考

### aspect_ratio 宽高比

| 值 | 说明 | 适用 |
|----|------|------|
| `"16:9"` | 横屏（默认） | 图片 + 视频 |
| `"9:16"` | 竖屏 | 图片 + 视频 |
| `"1:1"` | 方图 | 仅图片 |
| `"4:3"` | 横屏 4:3 | 仅图片 |
| `"3:4"` | 竖屏 3:4 | 仅图片 |
| `"landscape"` | 横屏（别名） | 图片 + 视频 |
| `"portrait"` | 竖屏（别名） | 图片 + 视频 |
| `"square"` | 方图（别名） | 仅图片 |

### resolution 分辨率

#### 图片分辨率

| 值 | 说明 | 支持的模型 |
|----|------|-----------|
| `null` / `"1K"` | 标准分辨率（默认） | 所有图片模型 |
| `"2K"` | 2K 高清 | GEM_PIX_2, NARWHAL |
| `"4K"` | 4K 超高清 | GEM_PIX_2, NARWHAL |

#### 视频分辨率

| 值 | 说明 | 备注 |
|----|------|------|
| `null` | 标准分辨率（默认） | - |
| `"1080p"` | 1080P 放大 | 仅 Veo 3.1 支持，可能需要 30 分钟 |
| `"4k"` | 4K 放大 | 仅 Veo 3.1 支持，可能需要 30 分钟 |

### response_format 响应格式

仅在 Gemini 原生格式中生效，控制图片的返回方式。

| 值 | 说明 |
|----|------|
| `"b64_json"` | 返回 Base64 编码的图片数据（默认） |
| `"url"` | 返回图片 URL |

在 OpenAI 格式中，通过请求体的 `response_format` 字段传递：

```json
{"response_format": "b64_json"}
```

在 Gemini 原生格式中，通过 `imageConfig.outputFormat` 传递：

```json
{"generationConfig": {"imageConfig": {"outputFormat": "b64_json"}}}
```

---

## 错误码

| HTTP 状态码 | 说明 |
|------------|------|
| `200` | 请求成功 |
| `400` | 请求参数错误（模型不支持、提示词为空等） |
| `401` | 认证失败（API Key 无效） |
| `500` | 服务端错误（生成失败、Token 不可用等） |

**错误响应示例**

```json
{
  "detail": "不支持的模型: unknown-model"
}
```

Gemini 格式错误响应：

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [{"text": "Error: 生成失败"}]
      },
      "finishReason": "ERROR"
    }
  ]
}
```
