# 远程有头打码服务 (Remote Browser Captcha Service)

为 flow2api 提供独立部署的 reCAPTCHA v3 Enterprise 打码服务，支持 **Token 预热池**，实现毫秒级响应。

---

## 目录

- [架构概览](#架构概览)
- [快速部署](#快速部署)
- [环境变量](#环境变量)
- [flow2api 配置](#flow2api-配置)
- [API 接口](#api-接口)
- [Token 预热池](#token-预热池)
- [性能测试](#性能测试)
- [常见问题](#常见问题)

---

## 架构概览

```
flow2api (任意环境)               远程打码服务 (有桌面/Xvfb 的机器)
┌──────────────────┐             ┌────────────────────────────────┐
│                  │             │                                │
│  captcha_method  │             │  ┌──────────┐  ┌────────────┐ │
│  = remote_browser│──── HTTP ──►│  │ Token 池  │  │ Playwright │ │
│                  │             │  │ (预热缓存) │◄─│ + Chromium │ │
│                  │◄── token ──│  └──────────┘  └────────────┘ │
│                  │  (<10ms)    │                                │
└──────────────────┘             └────────────────────────────────┘
```

**工作流程：**

1. 服务启动后，后台 `POOL_WORKERS` 个浏览器持续预打码，维持池中 `POOL_SIZE` 个可用 token
2. flow2api 调用 `POST /api/v1/solve` 时，优先从池中取 token（毫秒级返回）
3. 池空时自动降级为实时打码（~12-15 秒）
4. 过期 token 自动淘汰（`TOKEN_TTL` 秒），后台持续补充

---

## 快速部署

### 方式一：Docker 部署（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，至少修改 API_KEY

# 2. 构建并启动
docker compose up -d --build

# 3. 查看日志
docker compose logs -f

# 4. 健康检查
curl http://localhost:8060/health
```

### 方式二：本地部署

> **前提：** 需要桌面环境（Windows/macOS 或 Linux + Xvfb）

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 安装 Chromium 浏览器
python -m playwright install chromium

# 3. 设置环境变量
```

**Windows PowerShell:**
```powershell
$env:API_KEY = "your_secret_key"
$env:POOL_SIZE = "10"
$env:MAX_BROWSERS = "10"
python server.py
```

**Linux / macOS:**
```bash
export API_KEY="your_secret_key"
export POOL_SIZE="10"
export MAX_BROWSERS="10"
python server.py
```

启动后看到以下日志即为成功：
```
🚀 远程有头打码服务启动 | 端口: 8060 | 最大并发: 10
🔥 Token 预热池启动 | 目标: 10 | TTL: 90s | 并发: 3 | 项目: default_project
```

---

## 环境变量

### 基础配置

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `API_KEY` | `fcs_default_key` | API 认证密钥，**生产环境必须修改** |
| `PORT` | `8060` | 服务监听端口 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `MAX_BROWSERS` | `2` | 最大同时运行的浏览器数量（含预热和实时打码） |
| `SESSION_TIMEOUT` | `1800` | Session 过期清理时间（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG / INFO / WARNING / ERROR） |

### 代理配置

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `BROWSER_PROXY_URL` | 空 | 浏览器全局代理，支持 HTTP 和 SOCKS5 |

代理格式示例：
```
http://user:pass@proxy.com:8080
socks5://proxy.com:1080
http://proxy.com:3128
```

> ⚠️ SOCKS5 代理不支持用户名密码认证（Chromium 限制），会自动降级为 HTTP

### Token 预热池配置

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `POOL_ENABLED` | `true` | 是否启用 Token 预热池 |
| `POOL_SIZE` | `10` | 目标维持的可用 token 数量 |
| `TOKEN_TTL` | `90` | Token 有效期（秒）。reCAPTCHA token 约 120 秒有效，默认 90 秒提前淘汰 |
| `POOL_WORKERS` | `3` | 同时运行的预热浏览器数量 |
| `POOL_CHECK_INTERVAL` | `5` | 水位检查间隔（秒） |
| `POOL_DEFAULT_PROJECT` | `default_project` | 预热打码使用的 project_id |

**参数调优建议：**

- `POOL_SIZE` 应 ≥ 预期的突发并发量，建议与 `MAX_BROWSERS` 保持一致
- `TOKEN_TTL` 越小 token 越新鲜，但浪费更多；越大过期风险越高。建议保持 60-100 秒
- `POOL_WORKERS` 控制补充速度，建议 ≤ `MAX_BROWSERS` 的一半，避免占满浏览器资源
- `MAX_BROWSERS` 决定总并发上限（预热 + 实时降级），每个浏览器约占 **200-400MB 内存**

---

## flow2api 配置

在 flow2api 管理后台 → **系统配置** → **验证码配置**：

| 配置项 | 值 | 说明 |
|-------|---|------|
| 打码方式 | `远程有头打码` | 选择 remote_browser |
| 远程服务 Base URL | `http://<打码服务器IP>:8060` | 打码服务地址 |
| 远程服务 API Key | 与 `.env` 中 `API_KEY` 一致 | 认证密钥 |
| 远程请求超时（秒） | `60` | 图片生成超时，视频会自动调整 |

配置完成后，点击 **"测试当前打码分数"** 按钮验证连通性。

---

## API 接口

所有接口（除 `/health` 和 `/pool/status`）均需要 Bearer Token 认证：

```
Authorization: Bearer <API_KEY>
```

### `GET /health` — 健康检查

无需认证。

```bash
curl http://localhost:8060/health
```

**响应示例：**
```json
{
  "status": "ok",
  "active_sessions": 3,
  "max_browsers": 10,
  "pool_available": 8,
  "pool_target": 10
}
```

---

### `GET /pool/status` — 预热池状态

无需认证。返回预热池详细状态，包括每个 token 的年龄和命中率统计。

```bash
curl http://localhost:8060/pool/status
```

**响应示例：**
```json
{
  "enabled": true,
  "pool_size_target": 10,
  "available": 10,
  "total_in_pool": 10,
  "token_ttl": 90,
  "pool_workers": 3,
  "stats": {
    "total_produced": 30,
    "total_served": 20,
    "total_expired": 6,
    "pool_hits": 20,
    "pool_misses": 0,
    "hit_rate": "100.0%"
  },
  "tokens": [
    {"age_s": 12.3, "action": "IMAGE_GENERATION", "expired": false},
    {"age_s": 25.1, "action": "IMAGE_GENERATION", "expired": false}
  ],
  "avg_age_s": 18.7
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `available` | 当前可用（未过期）的 token 数量 |
| `pool_size_target` | 目标池大小 |
| `stats.pool_hits` | 池命中次数（毫秒级返回） |
| `stats.pool_misses` | 池未命中次数（降级实时打码） |
| `stats.hit_rate` | 池命中率 |
| `stats.total_expired` | 因超过 TTL 被丢弃的 token 数量 |

---

### `POST /api/v1/solve` — 获取 reCAPTCHA Token

核心接口。优先从预热池取 token（<10ms），池空时降级为实时打码（~12-15s）。

```bash
curl -X POST http://localhost:8060/api/v1/solve \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "your_project_id",
    "action": "IMAGE_GENERATION"
  }'
```

**请求参数：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|-------|------|
| `project_id` | string | ✅ | — | Flow 项目 ID |
| `action` | string | ❌ | `IMAGE_GENERATION` | reCAPTCHA action，可选 `IMAGE_GENERATION`、`VIDEO_GENERATION` 等 |
| `token_id` | int | ❌ | — | Token 标识（日志用） |
| `proxy_url` | string | ❌ | — | 本次请求使用的代理（覆盖全局代理） |

**响应示例：**
```json
{
  "token": "0cAFcWeA6TJmZw-tSPqW...",
  "session_id": "a1b2c3d4e5f6...",
  "fingerprint": {
    "user_agent": "Mozilla/5.0 ...",
    "accept_language": "en-US",
    "sec_ch_ua": "\"Chromium\";v=\"132\"...",
    "sec_ch_ua_mobile": "?0",
    "sec_ch_ua_platform": "\"Windows\""
  },
  "pool_hit": true
}
```

**响应字段说明：**

| 字段 | 说明 |
|------|------|
| `token` | reCAPTCHA v3 Enterprise token |
| `session_id` | 会话 ID，用于 finish/error 回调 |
| `fingerprint` | 浏览器指纹，flow2api 用于构造请求头 |
| `pool_hit` | `true` = 从预热池取出（毫秒级），`false` = 实时打码 |

---

### `POST /api/v1/sessions/{session_id}/finish` — 请求完成回调

flow2api 在图片/视频生成请求成功后自动调用，无需手动操作。

```bash
curl -X POST http://localhost:8060/api/v1/sessions/{session_id}/finish \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"status": "success"}'
```

---

### `POST /api/v1/sessions/{session_id}/error` — 错误回调

flow2api 在遇到 403 或 reCAPTCHA 验证失败时自动调用，用于记录错误日志。

```bash
curl -X POST http://localhost:8060/api/v1/sessions/{session_id}/error \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"error_reason": "recaptcha_failed"}'
```

---

### `POST /api/v1/custom-score` — 分数测试

flow2api 管理后台"测试当前打码分数"功能调用此接口。在真实测试页面（antcpt.com）上执行 reCAPTCHA 并从 DOM 读取分数。

```bash
curl -X POST http://localhost:8060/api/v1/custom-score \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "website_url": "https://antcpt.com/score_detector/",
    "website_key": "6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf",
    "action": "homepage",
    "verify_url": "https://antcpt.com/score_detector/verify.php",
    "enterprise": false
  }'
```

> ⚠️ 分数测试不走预热池，每次实时打码，耗时约 20-30 秒（含 12 秒页面预热）

---

## Token 预热池

### 工作原理

1. **启动阶段：** 服务启动 2 秒后开始预热，`POOL_WORKERS` 个浏览器并发打码
2. **补充循环：** 每 `POOL_CHECK_INTERVAL` 秒检查池水位，不足时自动并发补充
3. **取用逻辑：** `/api/v1/solve` 优先从池中取 token，匹配 action 优先
4. **过期淘汰：** 超过 `TOKEN_TTL` 秒的 token 自动丢弃
5. **降级保障：** 池空时自动降级为实时打码，不会丢失请求

### 预热时间线示例

```
t=0s    服务启动
t=2s    开始首轮预热（3 个浏览器并发）
t=14s   首批 3 个 token 入池
t=19s   第 2 批入池（6 个可用）
t=24s   第 3 批入池（9 个可用）
t=29s   第 4 批入池（10 个可用，满池！）
t=92s   首批 token 过期，自动补充...
```

### 禁用预热池

如果不需要预热池（例如低流量场景），设置：

```bash
POOL_ENABLED=false
```

此时 `/api/v1/solve` 每次都会实时打码，行为与预热池功能添加前一致。

---

## 性能测试

使用 `test_pool_perf.py` 进行 10 并发基准测试。

### 测试结果

| 模式 | 总耗时 | 平均耗时 | 最快 | 最慢 | 吞吐量 |
|------|-------|---------|------|------|-------|
| 无池 (MAX_BROWSERS=2) | 52.62s | 31.29s | 10.22s | 52.61s | 0.19 req/s |
| 无池 (MAX_BROWSERS=10) | 13.47s | 12.95s | 12.36s | 13.46s | 0.74 req/s |
| **有池 (POOL_SIZE=10)** | **0.01s** | **4.9ms** | **2.1ms** | **7.1ms** | **~1000 req/s** |

> 预热池命中时，响应速度提升约 **2,500 倍**

### 运行测试

```bash
# 基础性能测试（10 并发 /api/v1/solve）
python test_perf.py

# 预热池性能测试（等待池满后测试）
python test_pool_perf.py

# 查看池状态
python check_pool.py
```

---

## 常见问题

### 1. Docker 容器中 Chromium 崩溃

确保设置 `shm_size: "2gb"`，默认的 64MB 共享内存不够 Chromium 使用：

```yaml
services:
  captcha-service:
    shm_size: "2gb"
```

### 2. 服务需要访问哪些域名？

以下域名必须可达（不可被防火墙拦截）：
- `google.com` — reCAPTCHA 服务
- `gstatic.com` — reCAPTCHA 静态资源
- `recaptcha.net` — reCAPTCHA 备用域名
- `labs.google` — Google Labs 页面

### 3. 内存占用估算

每个浏览器实例约占 200-400MB 内存：

| MAX_BROWSERS | 预估内存 |
|-------------|---------|
| 2 | 0.4 - 0.8 GB |
| 5 | 1.0 - 2.0 GB |
| 10 | 2.0 - 4.0 GB |

### 4. 预热池 token 过期浪费

如果流量很低，池中 token 会在 `TOKEN_TTL` 过后过期被丢弃。可以：
- 减小 `POOL_SIZE`（如设为 2-3）
- 增大 `TOKEN_TTL`（如设为 110，但有过期下发风险）
- 设置 `POOL_ENABLED=false` 禁用预热池

### 5. 如何监控服务状态？

```bash
# 健康检查（含池可用数）
curl http://localhost:8060/health

# 预热池详细状态（含命中率、每个 token 年龄）
curl http://localhost:8060/pool/status
```
