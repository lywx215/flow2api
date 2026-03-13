"""
远程有头打码服务 (Remote Browser Captcha Service)
独立部署的 FastAPI 服务，为 flow2api 提供 reCAPTCHA v3 Enterprise token
"""
import os
import sys
import subprocess
import asyncio
import time
import random
import uuid
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ==================== 配置 ====================
API_KEY = os.environ.get("API_KEY", "fcs_default_key")
PORT = int(os.environ.get("PORT", "8060"))
HOST = os.environ.get("HOST", "0.0.0.0")
BROWSER_PROXY_URL = os.environ.get("BROWSER_PROXY_URL", "")
MAX_BROWSERS = int(os.environ.get("MAX_BROWSERS", "2"))
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "1800"))  # 30min
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# 代理池配置（优先级高于 BROWSER_PROXY_URL）
PROXY_POOL_API_URL = os.environ.get("PROXY_POOL_API_URL", "")
PROXY_POOL_REFRESH_INTERVAL = int(os.environ.get("PROXY_POOL_REFRESH_INTERVAL", "300"))  # 5min

# Token 预热池配置
POOL_SIZE = int(os.environ.get("POOL_SIZE", "10"))          # 目标维持的可用 token 数
TOKEN_TTL = int(os.environ.get("TOKEN_TTL", "90"))          # token 有效期(秒)，reCAPTCHA ~120s
POOL_WORKERS = int(os.environ.get("POOL_WORKERS", "3"))     # 同时补充的浏览器数
POOL_CHECK_INTERVAL = int(os.environ.get("POOL_CHECK_INTERVAL", "5"))  # 水位检查间隔(秒)
POOL_COOLDOWN = int(os.environ.get("POOL_COOLDOWN", "3"))              # 每批补充后冷却(秒)
POOL_BROWSER_MAX_USES = int(os.environ.get("POOL_BROWSER_MAX_USES", "50"))  # 每个浏览器最大复用次数
POOL_DEFAULT_PROJECT = os.environ.get("POOL_DEFAULT_PROJECT", "default_project")
POOL_ENABLED = os.environ.get("POOL_ENABLED", "true").lower() in ("true", "1", "yes")

# reCAPTCHA 配置
WEBSITE_KEY = "6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV"
LABS_URL_TEMPLATE = "https://labs.google/fx/tools/flow/project/{project_id}"

# ==================== 日志 ====================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("captcha_service")

# ==================== Playwright 安装 ====================
def ensure_playwright():
    """确保 playwright 和 chromium 已安装"""
    try:
        import playwright
        logger.info("playwright 已安装")
    except ImportError:
        logger.info("playwright 未安装，正在安装...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "playwright>=1.40.0"],
            check=True, timeout=300,
        )
        logger.info("playwright 安装成功")

    # 检查 chromium 是否已安装
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            path = p.chromium.executable_path
            if path and os.path.exists(path):
                logger.info(f"chromium 已安装: {path}")
                return
    except Exception:
        pass

    logger.info("正在安装 chromium...")
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True, timeout=600,
    )
    logger.info("chromium 安装成功")


ensure_playwright()
from playwright.async_api import async_playwright

# ==================== UA 池 ====================
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.210 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.210 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

RESOLUTIONS = [
    (1920, 1080), (2560, 1440), (1366, 768), (1536, 864),
    (1600, 900), (1280, 720), (1440, 900), (1680, 1050),
]


# ==================== 代理解析 ====================
def parse_proxy_url(proxy_url: str) -> Optional[Dict[str, str]]:
    """解析代理 URL 为 Playwright 格式"""
    if not proxy_url:
        return None
    import re
    if not re.match(r'^(http|https|socks5)://', proxy_url):
        proxy_url = f"http://{proxy_url}"
    match = re.match(r'^(socks5|http|https)://(?:([^:]+):([^@]+)@)?([^:]+):(\d+)$', proxy_url)
    if match:
        protocol, username, password, host, port = match.groups()
        # Chromium 不支持带认证的 socks5，自动降级为 http
        if protocol == "socks5" and username and password:
            protocol = "http"
            logger.warning(f"SOCKS5 代理不支持认证，已降级为 HTTP: http://{host}:{port}")
        proxy_config = {'server': f'{protocol}://{host}:{port}'}
        if username and password:
            proxy_config['username'] = username
            proxy_config['password'] = password
        return proxy_config
    return None


# ==================== 代理池管理 ====================
class ProxyPool:
    """住宅代理池：从 API 批量获取旋转代理，为每个浏览器分配独立代理"""

    def __init__(self, api_url: str):
        self._api_url = api_url
        self._proxies: List[str] = []  # 格式: http://user:pass@host:port
        self._index = 0
        self._lock = asyncio.Lock()
        self._last_refresh = 0.0
        self._refresh_interval = PROXY_POOL_REFRESH_INTERVAL

    @staticmethod
    def parse_proxy_line(line: str) -> Optional[str]:
        """解析代理行: host:port:user:pass -> http://user:pass@host:port"""
        line = line.strip()
        if not line:
            return None
        parts = line.split(":")
        if len(parts) == 4:
            host, port, user, password = parts
            return f"http://{user}:{password}@{host}:{port}"
        elif len(parts) == 2:
            return f"http://{parts[0]}:{parts[1]}"
        elif len(parts) == 3:
            return f"http://{parts[2]}@{parts[0]}:{parts[1]}"
        return None

    async def refresh(self):
        """从 API 刷新代理列表"""
        try:
            import urllib.request
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(self._api_url, timeout=15).read().decode("utf-8")
            )
            lines = resp.strip().replace("\r\n", "\n").split("\n")
            parsed = []
            for line in lines:
                proxy_url = self.parse_proxy_line(line)
                if proxy_url:
                    parsed.append(proxy_url)

            if parsed:
                async with self._lock:
                    self._proxies = parsed
                    self._index = 0
                    self._last_refresh = time.time()
                logger.info(f"[ProxyPool] ✅ 刷新成功: {len(parsed)} 个代理")
            else:
                logger.warning("[ProxyPool] ⚠️ API 返回空代理列表")
        except Exception as e:
            logger.error(f"[ProxyPool] ❌ 刷新失败: {type(e).__name__}: {str(e)[:200]}")

    async def get_proxy(self) -> Optional[str]:
        """获取一个代理 URL (轮询分配)"""
        if time.time() - self._last_refresh > self._refresh_interval:
            await self.refresh()

        async with self._lock:
            if not self._proxies:
                return None
            proxy = self._proxies[self._index % len(self._proxies)]
            self._index += 1
            return proxy

    @property
    def available(self) -> int:
        return len(self._proxies)

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._api_url),
            "available": len(self._proxies),
            "last_refresh": self._last_refresh,
            "refresh_interval": self._refresh_interval,
        }


# 全局代理池实例
proxy_pool = ProxyPool(PROXY_POOL_API_URL) if PROXY_POOL_API_URL else None


# ==================== Session 管理 ====================
class CaptchaSession:
    """一个打码会话，跟踪浏览器生命周期"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.token: Optional[str] = None
        self.fingerprint: Optional[Dict[str, Any]] = None
        self.finished = False
        self.error: Optional[str] = None


class SessionManager:
    """管理所有活跃的打码会话"""

    def __init__(self):
        self._sessions: Dict[str, CaptchaSession] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(MAX_BROWSERS)
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup_loop(self):
        """启动定时清理过期 session 的后台任务"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """定期清理超时的 session"""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            async with self._lock:
                expired = [
                    sid for sid, s in self._sessions.items()
                    if now - s.created_at > SESSION_TIMEOUT
                ]
                for sid in expired:
                    del self._sessions[sid]
                    logger.info(f"Session {sid[:8]} 已过期清理")

    async def create_session(self) -> CaptchaSession:
        session_id = uuid.uuid4().hex
        session = CaptchaSession(session_id)
        async with self._lock:
            self._sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[CaptchaSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def remove_session(self, session_id: str):
        async with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


session_manager = SessionManager()


# ==================== 持久化浏览器池 ====================
class PersistentBrowser:
    """常驻浏览器：启动一次，重复调用 execute() 生产 token"""

    def __init__(self, browser_id: int):
        self.browser_id = browser_id
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._ready = False
        self._uses = 0
        self._lock = asyncio.Lock()
        self._fingerprint: Optional[Dict[str, Any]] = None

    @property
    def is_ready(self) -> bool:
        return self._ready and self._page is not None

    @property
    def needs_restart(self) -> bool:
        return self._uses >= POOL_BROWSER_MAX_USES

    async def start(self):
        """启动浏览器并加载 enterprise.js（一次性）"""
        await self.close()
        try:
            user_agent = random.choice(UA_LIST)
            width, height = random.choice(RESOLUTIONS)
            height = height - random.randint(0, 80)
            viewport = {"width": width, "height": height}

            # 代理池优先：每个浏览器获取独立代理 IP
            effective_proxy = None
            if proxy_pool:
                effective_proxy = await proxy_pool.get_proxy()
                if effective_proxy:
                    logger.info(f"[PB-{self.browser_id}] 使用代理池代理: {effective_proxy[:50]}...")
            if not effective_proxy:
                effective_proxy = BROWSER_PROXY_URL
            proxy_option = parse_proxy_url(effective_proxy) if effective_proxy else None

            self._playwright = await async_playwright().start()
            browser_args = [
                '--disable-blink-features=AutomationControlled',
                '--disable-quic',
                '--disable-features=UseDnsHttpsSvcb',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-setuid-sandbox',
                '--no-first-run',
                '--no-zygote',
                f'--window-size={width},{height}',
                '--disable-infobars',
                '--hide-scrollbars',
                '--start-minimized',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--disable-gpu',
                '--disable-gpu-compositing',
                '--disable-software-rasterizer',
                '--disable-extensions',
                '--disable-default-apps',
                '--disable-sync',
                '--disable-translate',
                '--disable-background-networking',
                '--metrics-recording-only',
                '--mute-audio',
                '--no-default-browser-check',
            ]
            if sys.platform.startswith("win"):
                browser_args.append('--window-position=-32000,-32000')

            self._browser = await self._playwright.chromium.launch(
                headless=False,
                proxy=proxy_option,
                args=browser_args,
            )
            self._context = await self._browser.new_context(
                viewport=viewport,
                locale="en-US",
            )
            self._page = await self._context.new_page()
            await self._page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )

            using_proxy = bool(proxy_option)
            primary_host = "https://www.recaptcha.net" if using_proxy else "https://www.google.com"
            secondary_host = "https://www.google.com" if primary_host == "https://www.recaptcha.net" else "https://www.recaptcha.net"
            page_url = LABS_URL_TEMPLATE.format(project_id=POOL_DEFAULT_PROJECT)

            # 路由拦截
            async def handle_route(route):
                if route.request.url.rstrip('/') == page_url.rstrip('/'):
                    html = f"""<html><head><script>
                    (() => {{
                        const urls = [
                            '{primary_host}/recaptcha/enterprise.js?render={WEBSITE_KEY}',
                            '{secondary_host}/recaptcha/enterprise.js?render={WEBSITE_KEY}'
                        ];
                        const loadScript = (index) => {{
                            if (index >= urls.length) return;
                            const script = document.createElement('script');
                            script.src = urls[index];
                            script.async = true;
                            script.onerror = () => loadScript(index + 1);
                            document.head.appendChild(script);
                        }};
                        loadScript(0);
                    }})();
                    </script></head><body></body></html>"""
                    await route.fulfill(status=200, content_type="text/html", body=html)
                elif any(d in route.request.url for d in ["google.com", "gstatic.com", "recaptcha.net"]):
                    await route.continue_()
                else:
                    await route.abort()

            await self._page.route("**/*", handle_route)

            # 导航并等待 grecaptcha 就绪
            await self._page.goto(page_url, wait_until="load", timeout=30000)
            await self._page.wait_for_function(
                "typeof grecaptcha !== 'undefined' && typeof grecaptcha.enterprise !== 'undefined'",
                timeout=15000,
            )

            # 提取指纹
            self._fingerprint = await _capture_page_fingerprint(self._page, f"pb-{self.browser_id}")

            self._ready = True
            self._uses = 0
            logger.info(f"[PB-{self.browser_id}] ✅ 浏览器就绪")
        except Exception as e:
            logger.error(f"[PB-{self.browser_id}] 启动失败: {type(e).__name__}: {str(e)[:200]}")
            await self.close()
            raise

    async def produce_token(self, action: str = "IMAGE_GENERATION") -> Optional[Dict[str, Any]]:
        """在已加载的页面上调用 execute() 获取 token，并等待 reload/clr 回调确认"""
        async with self._lock:
            if not self.is_ready:
                return None
            try:
                # 设置 reload/clr 回调事件（每次 execute 都需要重新监听）
                reload_ok = asyncio.Event()
                clr_ok = asyncio.Event()

                def _on_response(response):
                    try:
                        if response.status != 200:
                            return
                        parsed = urlparse(response.url)
                        path = parsed.path or ""
                        if "recaptcha/enterprise/reload" not in path and "recaptcha/enterprise/clr" not in path:
                            return
                        query = parse_qs(parsed.query or "")
                        key = (query.get("k") or [None])[0]
                        if key != WEBSITE_KEY:
                            return
                        if "recaptcha/enterprise/reload" in path:
                            reload_ok.set()
                        elif "recaptcha/enterprise/clr" in path:
                            clr_ok.set()
                    except Exception:
                        pass

                self._page.on("response", _on_response)

                try:
                    # 执行 reCAPTCHA
                    token = await asyncio.wait_for(
                        self._page.evaluate(f"""
                            (actionName) => {{
                                return new Promise((resolve, reject) => {{
                                    const timeout = setTimeout(() => reject(new Error('timeout')), 25000);
                                    grecaptcha.enterprise.execute('{WEBSITE_KEY}', {{action: actionName}})
                                        .then(t => {{ clearTimeout(timeout); resolve(t); }})
                                        .catch(e => {{ clearTimeout(timeout); reject(e); }});
                                }});
                            }}
                        """, action),
                        timeout=30,
                    )

                    # 等待 enterprise/reload 回调（Google 服务端注册 token）
                    try:
                        await asyncio.wait_for(reload_ok.wait(), timeout=12)
                    except asyncio.TimeoutError:
                        logger.warning(f"[PB-{self.browser_id}] 等待 enterprise/reload 超时, token 可能无效")
                        self._ready = False
                        return None

                    # 等待 enterprise/clr 回调（Google 服务端确认）
                    try:
                        await asyncio.wait_for(clr_ok.wait(), timeout=12)
                    except asyncio.TimeoutError:
                        logger.warning(f"[PB-{self.browser_id}] 等待 enterprise/clr 超时, token 可能无效")
                        self._ready = False
                        return None

                    # 额外稳定等待
                    await asyncio.sleep(2)

                    self._uses += 1
                    logger.info(f"[PB-{self.browser_id}] token #{self._uses} 生成成功 (reload+clr 已确认)")
                    return {
                        "token": token,
                        "fingerprint": self._fingerprint,
                        "session_id": uuid.uuid4().hex,
                    }
                finally:
                    # 移除临时监听器，避免内存泄漏
                    try:
                        self._page.remove_listener("response", _on_response)
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"[PB-{self.browser_id}] execute 失败: {type(e).__name__}: {str(e)[:100]}")
                self._ready = False
                return None

    async def close(self):
        """关闭浏览器"""
        self._ready = False
        for obj, name in [
            (self._page, "page"), (self._context, "context"),
            (self._browser, "browser"), (self._playwright, "playwright"),
        ]:
            if obj:
                try:
                    close_fn = getattr(obj, "close", None) or getattr(obj, "stop", None)
                    if close_fn:
                        await asyncio.wait_for(close_fn(), timeout=10)
                except Exception:
                    pass
        self._page = self._context = self._browser = self._playwright = None


class BrowserPool:
    """管理 N 个持久化浏览器，轮询分配打码任务"""

    def __init__(self, size: int):
        self._browsers: List[PersistentBrowser] = [
            PersistentBrowser(i) for i in range(size)
        ]
        self._robin_index = 0
        self._lock = asyncio.Lock()

    async def start_all(self):
        """启动所有浏览器"""
        logger.info(f"🌐 启动 {len(self._browsers)} 个持久化浏览器...")
        for pb in self._browsers:
            try:
                await pb.start()
            except Exception as e:
                logger.error(f"[BrowserPool] 浏览器 {pb.browser_id} 启动失败: {e}")
            # 间隔启动，避免 CPU 峰值
            await asyncio.sleep(2)
        ready = sum(1 for pb in self._browsers if pb.is_ready)
        logger.info(f"🌐 持久化浏览器就绪: {ready}/{len(self._browsers)}")

    async def produce_token(self, action: str = "IMAGE_GENERATION") -> Optional[Dict[str, Any]]:
        """轮询选一个浏览器生产 token"""
        async with self._lock:
            start_idx = self._robin_index
            self._robin_index = (self._robin_index + 1) % len(self._browsers)

        # 尝试所有浏览器
        for offset in range(len(self._browsers)):
            idx = (start_idx + offset) % len(self._browsers)
            pb = self._browsers[idx]

            # 需要重启？
            if pb.needs_restart or not pb.is_ready:
                try:
                    logger.info(f"[BrowserPool] 重启浏览器 {pb.browser_id} (uses={pb._uses})")
                    await pb.start()
                except Exception:
                    continue

            result = await pb.produce_token(action)
            if result:
                return result

        return None

    async def close_all(self):
        for pb in self._browsers:
            await pb.close()

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "total": len(self._browsers),
            "ready": sum(1 for pb in self._browsers if pb.is_ready),
            "browsers": [
                {"id": pb.browser_id, "ready": pb.is_ready, "uses": pb._uses}
                for pb in self._browsers
            ],
        }


browser_pool = BrowserPool(POOL_WORKERS)


# ==================== Token 预热池 ====================
@dataclass
class PooledToken:
    """池中的一个预打码 token"""
    token: str
    fingerprint: Optional[Dict[str, Any]]
    session_id: str
    created_at: float
    action: str = "IMAGE_GENERATION"

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age > TOKEN_TTL


class TokenPool:
    """Token 预热池：通过持久化浏览器持续打码"""

    def __init__(self):
        self._tokens: List[PooledToken] = []
        self._lock = asyncio.Lock()
        self._replenish_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(POOL_WORKERS)
        self._total_produced = 0
        self._total_served = 0
        self._total_expired = 0
        self._pool_hits = 0
        self._pool_misses = 0
        self._started = False

    async def start(self):
        """启动预热池 + 持久化浏览器"""
        if not POOL_ENABLED:
            logger.info("🔴 Token 预热池已禁用 (POOL_ENABLED=false)")
            return
        # 先启动持久化浏览器
        await browser_pool.start_all()
        self._started = True
        self._replenish_task = asyncio.create_task(self._replenish_loop())
        logger.info(
            f"🔥 Token 预热池启动 | 目标: {POOL_SIZE} | TTL: {TOKEN_TTL}s | "
            f"持久化浏览器: {POOL_WORKERS} | 复用上限: {POOL_BROWSER_MAX_USES}"
        )

    async def get_token(self, action: str = "IMAGE_GENERATION") -> Optional[PooledToken]:
        """从池中取一个新鲜 token，无则返回 None"""
        async with self._lock:
            before = len(self._tokens)
            self._tokens = [t for t in self._tokens if not t.is_expired]
            expired_count = before - len(self._tokens)
            if expired_count > 0:
                self._total_expired += expired_count
                logger.info(f"[Pool] 清理 {expired_count} 个过期 token")

            for i, t in enumerate(self._tokens):
                if t.action == action:
                    self._tokens.pop(i)
                    self._pool_hits += 1
                    self._total_served += 1
                    logger.info(
                        f"[Pool] ✅ 命中! 池剩余: {len(self._tokens)}/{POOL_SIZE}, "
                        f"token age: {t.age:.1f}s"
                    )
                    return t

            if self._tokens:
                t = self._tokens.pop(0)
                self._pool_hits += 1
                self._total_served += 1
                logger.info(
                    f"[Pool] ⚠️ action 不匹配但有可用 token, 池剩余: {len(self._tokens)}/{POOL_SIZE}"
                )
                return t

            self._pool_misses += 1
            return None

    async def _replenish_loop(self):
        """主循环：检查水位 → 用持久化浏览器补充 token"""
        await asyncio.sleep(2)
        while True:
            try:
                await self._check_and_replenish()
            except Exception as e:
                logger.error(f"[Pool] 补充循环异常: {type(e).__name__}: {e}")
            await asyncio.sleep(POOL_CHECK_INTERVAL)

    async def _check_and_replenish(self):
        """检查水位并用持久化浏览器补充"""
        async with self._lock:
            before = len(self._tokens)
            self._tokens = [t for t in self._tokens if not t.is_expired]
            expired_count = before - len(self._tokens)
            if expired_count > 0:
                self._total_expired += expired_count
            current = len(self._tokens)

        deficit = POOL_SIZE - current
        if deficit <= 0:
            return

        logger.info(f"[Pool] 水位不足: {current}/{POOL_SIZE}, 需补充 {deficit} 个")

        added = 0
        failed = 0
        for _ in range(deficit):
            result = await browser_pool.produce_token("IMAGE_GENERATION")
            if result:
                pooled = PooledToken(
                    token=result["token"],
                    fingerprint=result.get("fingerprint"),
                    session_id=result["session_id"],
                    created_at=time.time(),
                    action="IMAGE_GENERATION",
                )
                async with self._lock:
                    self._tokens.append(pooled)
                    self._total_produced += 1
                added += 1
            else:
                failed += 1

            # 小间隔，避免 CPU 峰值
            if POOL_COOLDOWN > 0:
                await asyncio.sleep(POOL_COOLDOWN)

        if added > 0 or failed > 0:
            async with self._lock:
                pool_size = len(self._tokens)
            logger.info(f"[Pool] 补充完成: +{added} 成功, {failed} 失败, 当前: {pool_size}/{POOL_SIZE}")

    @property
    def status(self) -> Dict[str, Any]:
        fresh = [t for t in self._tokens if not t.is_expired]
        ages = [t.age for t in fresh] if fresh else []
        return {
            "enabled": POOL_ENABLED and self._started,
            "pool_size_target": POOL_SIZE,
            "available": len(fresh),
            "total_in_pool": len(self._tokens),
            "token_ttl": TOKEN_TTL,
            "pool_workers": POOL_WORKERS,
            "browser_pool": browser_pool.status,
            "stats": {
                "total_produced": self._total_produced,
                "total_served": self._total_served,
                "total_expired": self._total_expired,
                "pool_hits": self._pool_hits,
                "pool_misses": self._pool_misses,
                "hit_rate": f"{self._pool_hits / max(1, self._pool_hits + self._pool_misses) * 100:.1f}%",
            },
            "tokens": [
                {"age_s": round(t.age, 1), "action": t.action, "expired": t.is_expired}
                for t in self._tokens
            ],
            "avg_age_s": round(sum(ages) / len(ages), 1) if ages else 0,
        }


token_pool = TokenPool()


# ==================== 核心打码逻辑 ====================
async def solve_captcha(
    project_id: str,
    action: str = "IMAGE_GENERATION",
    proxy_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    启动有头浏览器，获取 reCAPTCHA v3 Enterprise token。

    核心逻辑从 browser_captcha.py 的 _execute_captcha 方法提取。
    """
    session = await session_manager.create_session()
    playwright_instance = None
    browser = None
    context = None

    try:
        # 随机选择 UA 和分辨率
        user_agent = random.choice(UA_LIST)
        width, height = random.choice(RESOLUTIONS)
        height = height - random.randint(0, 80)
        viewport = {"width": width, "height": height}

        # 解析代理
        effective_proxy = proxy_url or BROWSER_PROXY_URL
        proxy_option = parse_proxy_url(effective_proxy) if effective_proxy else None
        using_proxy = bool(proxy_option)

        logger.info(
            f"[{session.session_id[:8]}] 开始打码: project={project_id}, "
            f"action={action}, proxy={'yes' if using_proxy else 'no'}"
        )

        # 启动浏览器
        playwright_instance = await async_playwright().start()
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-quic',
            '--disable-features=UseDnsHttpsSvcb',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-setuid-sandbox',
            '--no-first-run',
            '--no-zygote',
            f'--window-size={width},{height}',
            '--disable-infobars',
            '--hide-scrollbars',
            '--start-minimized',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            # CPU 优化
            '--disable-gpu',
            '--disable-gpu-compositing',
            '--disable-software-rasterizer',
            '--disable-extensions',
            '--disable-default-apps',
            '--disable-sync',
            '--disable-translate',
            '--disable-background-networking',
            '--metrics-recording-only',
            '--mute-audio',
            '--no-default-browser-check',
        ]

        # Windows 下窗口最小化到屏幕外
        if sys.platform.startswith("win"):
            browser_args.append('--window-position=-32000,-32000')

        browser = await playwright_instance.chromium.launch(
            headless=False,
            proxy=proxy_option,
            args=browser_args,
        )
        context = await browser.new_context(
            viewport=viewport,
            locale="en-US",
        )

        # 执行打码
        token, fingerprint = await _execute_captcha_in_context(
            context, project_id, action, using_proxy, session.session_id
        )

        if not token:
            raise RuntimeError("打码失败：未获取到 reCAPTCHA token")

        session.token = token
        session.fingerprint = fingerprint

        logger.info(
            f"[{session.session_id[:8]}] ✅ 打码成功: token={token[:20]}..."
        )

        return {
            "token": token,
            "session_id": session.session_id,
            "fingerprint": fingerprint,
        }

    except Exception as e:
        session.error = str(e)
        logger.error(f"[{session.session_id[:8]}] ❌ 打码失败: {e}")
        raise
    finally:
        # 关闭浏览器
        try:
            if context:
                await asyncio.wait_for(context.close(), timeout=10)
        except Exception:
            pass
        try:
            if browser:
                await asyncio.wait_for(browser.close(), timeout=10)
        except Exception:
            pass
        try:
            if playwright_instance:
                await asyncio.wait_for(playwright_instance.stop(), timeout=10)
        except Exception:
            pass


async def _execute_captcha_in_context(
    context,
    project_id: str,
    action: str,
    using_proxy: bool,
    session_id: str,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """在浏览器上下文中执行打码"""
    page = None
    try:
        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        page_url = LABS_URL_TEMPLATE.format(project_id=project_id)
        primary_host = "https://www.recaptcha.net" if using_proxy else "https://www.google.com"
        secondary_host = "https://www.google.com" if primary_host == "https://www.recaptcha.net" else "https://www.recaptcha.net"

        logger.info(
            f"[{session_id[:8]}] 加载 enterprise.js: "
            f"primary={primary_host}, secondary={secondary_host}"
        )

        # 路由拦截：只放行 reCAPTCHA 相关请求
        async def handle_route(route):
            if route.request.url.rstrip('/') == page_url.rstrip('/'):
                html = f"""<html><head><script>
                (() => {{
                    const urls = [
                        '{primary_host}/recaptcha/enterprise.js?render={WEBSITE_KEY}',
                        '{secondary_host}/recaptcha/enterprise.js?render={WEBSITE_KEY}'
                    ];
                    const loadScript = (index) => {{
                        if (index >= urls.length) return;
                        const script = document.createElement('script');
                        script.src = urls[index];
                        script.async = true;
                        script.onerror = () => loadScript(index + 1);
                        document.head.appendChild(script);
                    }};
                    loadScript(0);
                }})();
                </script></head><body></body></html>"""
                await route.fulfill(status=200, content_type="text/html", body=html)
            elif any(d in route.request.url for d in ["google.com", "gstatic.com", "recaptcha.net"]):
                await route.continue_()
            else:
                await route.abort()

        def handle_request_failed(request):
            try:
                failed_url = request.url or ""
                if not any(d in failed_url for d in ["google.com", "gstatic.com", "recaptcha.net"]):
                    return
                failure = request.failure or ""
                logger.warning(f"[{session_id[:8]}] 资源加载失败: url={failed_url[:200]}, error={failure}")
            except Exception:
                pass

        await page.route("**/*", handle_route)
        page.on("requestfailed", handle_request_failed)

        # 监听 enterprise/reload 和 enterprise/clr 响应
        reload_ok_event = asyncio.Event()
        clr_ok_event = asyncio.Event()

        def handle_response(response):
            try:
                if response.status != 200:
                    return
                parsed = urlparse(response.url)
                path = parsed.path or ""
                if "recaptcha/enterprise/reload" not in path and "recaptcha/enterprise/clr" not in path:
                    return
                query = parse_qs(parsed.query or "")
                key = (query.get("k") or [None])[0]
                if key != WEBSITE_KEY:
                    return
                if "recaptcha/enterprise/reload" in path:
                    reload_ok_event.set()
                elif "recaptcha/enterprise/clr" in path:
                    clr_ok_event.set()
            except Exception:
                pass

        page.on("response", handle_response)

        # 导航到页面
        try:
            await page.goto(page_url, wait_until="load", timeout=30000)
        except Exception as e:
            logger.warning(f"[{session_id[:8]}] page.goto 失败: {type(e).__name__}: {str(e)[:200]}")
            return None, None

        # 等待 grecaptcha 就绪
        try:
            await page.wait_for_function("typeof grecaptcha !== 'undefined'", timeout=15000)
        except Exception as e:
            logger.warning(f"[{session_id[:8]}] grecaptcha 未就绪: {type(e).__name__}: {str(e)[:200]}")
            return None, None

        # 提取浏览器指纹
        fingerprint = await _capture_page_fingerprint(page, session_id)

        # 执行 reCAPTCHA
        token = await asyncio.wait_for(
            page.evaluate(f"""
                (actionName) => {{
                    return new Promise((resolve, reject) => {{
                        const timeout = setTimeout(() => reject(new Error('timeout')), 25000);
                        grecaptcha.enterprise.execute('{WEBSITE_KEY}', {{action: actionName}})
                            .then(t => {{ clearTimeout(timeout); resolve(t); }})
                            .catch(e => {{ clearTimeout(timeout); reject(e); }});
                    }});
                }}
            """, action),
            timeout=30,
        )

        # 等待 enterprise/reload 请求完成
        try:
            await asyncio.wait_for(reload_ok_event.wait(), timeout=12)
        except asyncio.TimeoutError:
            logger.warning(f"[{session_id[:8]}] 等待 enterprise/reload 超时")
            return None, None

        # 等待 enterprise/clr 请求完成
        try:
            await asyncio.wait_for(clr_ok_event.wait(), timeout=12)
        except asyncio.TimeoutError:
            logger.warning(f"[{session_id[:8]}] 等待 enterprise/clr 超时")
            return None, None

        # 额外等待确保稳定
        await asyncio.sleep(3)

        return token, fingerprint

    except Exception as e:
        logger.warning(f"[{session_id[:8]}] 打码异常: {type(e).__name__}: {str(e)[:200]}")
        return None, None
    finally:
        if page:
            try:
                await page.close()
            except Exception:
                pass


async def _capture_page_fingerprint(page, session_id: str) -> Optional[Dict[str, Any]]:
    """从浏览器页面提取指纹信息"""
    try:
        fingerprint = await page.evaluate("""
            () => {
                const ua = navigator.userAgent || "";
                const lang = navigator.language || "";
                const uaData = navigator.userAgentData || null;
                let secChUa = "";
                let secChUaMobile = "";
                let secChUaPlatform = "";

                if (uaData) {
                    if (Array.isArray(uaData.brands) && uaData.brands.length > 0) {
                        secChUa = uaData.brands
                            .map((item) => `"${item.brand}";v="${item.version}"`)
                            .join(", ");
                    }
                    secChUaMobile = uaData.mobile ? "?1" : "?0";
                    if (uaData.platform) {
                        secChUaPlatform = `"${uaData.platform}"`;
                    }
                }

                return {
                    user_agent: ua,
                    accept_language: lang,
                    sec_ch_ua: secChUa,
                    sec_ch_ua_mobile: secChUaMobile,
                    sec_ch_ua_platform: secChUaPlatform,
                };
            }
        """)
        return fingerprint if isinstance(fingerprint, dict) else None
    except Exception as e:
        logger.warning(f"[{session_id[:8]}] 指纹提取失败: {e}")
        return None


# ==================== API 模型 ====================
class SolveRequest(BaseModel):
    project_id: str
    action: str = "IMAGE_GENERATION"
    token_id: Optional[int] = None
    proxy_url: Optional[str] = None


class CustomScoreRequest(BaseModel):
    website_url: str = "https://antcpt.com/score_detector/"
    website_key: str = "6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf"
    verify_url: str = "https://antcpt.com/score_detector/verify.php"
    action: str = "homepage"
    enterprise: bool = False


class FinishRequest(BaseModel):
    status: str = "success"


class ErrorRequest(BaseModel):
    error_reason: str = "unknown"


# ==================== 认证 ====================
security = HTTPBearer()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials


# ==================== FastAPI 应用 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    session_manager.start_cleanup_loop()
    await token_pool.start()
    logger.info(f"🚀 远程有头打码服务启动 | 端口: {PORT} | 最大并发: {MAX_BROWSERS}")
    if BROWSER_PROXY_URL:
        logger.info(f"📡 全局代理: {BROWSER_PROXY_URL}")
    yield
    await browser_pool.close_all()
    logger.info("🛑 服务关闭")


app = FastAPI(
    title="Remote Browser Captcha Service",
    description="为 flow2api 提供远程有头浏览器 reCAPTCHA 打码服务",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """健康检查"""
    pool_status = token_pool.status
    return {
        "status": "ok",
        "active_sessions": session_manager.active_count,
        "max_browsers": MAX_BROWSERS,
        "pool_available": pool_status["available"],
        "pool_target": pool_status["pool_size_target"],
    }


@app.get("/pool/status")
async def pool_status():
    """Token 预热池状态"""
    return token_pool.status


@app.post("/api/v1/solve")
async def solve(
    req: SolveRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    核心打码接口，获取 reCAPTCHA v3 Enterprise token。

    优先从预热池取 token（毫秒级），池空时降级为实时打码。
    """
    # 1. 尝试从预热池获取
    pooled = await token_pool.get_token(action=req.action)
    if pooled:
        # 创建 session 用于 finish/error 回调
        session = await session_manager.create_session()
        session.token = pooled.token
        session.fingerprint = pooled.fingerprint
        logger.info(
            f"[{session.session_id[:8]}] 🚀 池命中! age={pooled.age:.1f}s"
        )
        return {
            "token": pooled.token,
            "session_id": session.session_id,
            "fingerprint": pooled.fingerprint,
            "pool_hit": True,
        }

    # 2. 池空，降级为实时打码
    logger.info("[solve] 池空，降级实时打码")
    acquired = False
    try:
        acquired = session_manager._semaphore._value > 0
        if not acquired:
            logger.warning("打码队列已满，等待中...")
        await asyncio.wait_for(
            session_manager._semaphore.acquire(),
            timeout=120,
        )
        acquired = True
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=f"打码队列已满 (max={MAX_BROWSERS})，请稍后重试"
        )

    try:
        result = await solve_captcha(
            project_id=req.project_id,
            action=req.action,
            proxy_url=req.proxy_url,
        )
        result["pool_hit"] = False
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"打码异常: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"打码失败: {str(e)}")
    finally:
        if acquired:
            session_manager._semaphore.release()


@app.post("/api/v1/sessions/{session_id}/finish")
async def session_finish(
    session_id: str,
    req: FinishRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    请求结束回调，通知打码服务释放资源。

    flow2api 在图片/视频生成请求结束后调用。
    """
    session = await session_manager.get_session(session_id)
    if session:
        session.finished = True
        await session_manager.remove_session(session_id)
        logger.info(f"[{session_id[:8]}] 收到 finish 回调: status={req.status}")
    else:
        logger.info(f"[{session_id[:8]}] finish 回调 (session 已过期)")
    return {"success": True}


@app.post("/api/v1/sessions/{session_id}/error")
async def session_error(
    session_id: str,
    req: ErrorRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    错误回调，通知打码服务上游请求失败。

    flow2api 在遇到 403/reCAPTCHA 验证失败时调用。
    """
    session = await session_manager.get_session(session_id)
    if session:
        session.error = req.error_reason
        logger.warning(f"[{session_id[:8]}] 收到 error 回调: reason={req.error_reason}")
    else:
        logger.warning(f"[{session_id[:8]}] error 回调 (session 已过期): reason={req.error_reason}")
    return {"success": True}


@app.post("/api/v1/custom-score")
async def custom_score(
    req: CustomScoreRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    自定义站点打码 + 分数测试。

    flow2api 的"测试当前打码分数"功能调用此接口。
    在真实页面上执行 reCAPTCHA，并从页面 DOM 中读取分数。
    """
    acquired = False
    try:
        acquired = session_manager._semaphore._value > 0
        if not acquired:
            logger.warning("打码队列已满，等待中...")
        await asyncio.wait_for(
            session_manager._semaphore.acquire(),
            timeout=120,
        )
        acquired = True
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=f"打码队列已满 (max={MAX_BROWSERS})，请稍后重试"
        )

    playwright_instance = None
    browser = None
    context = None

    try:
        token_start = time.time()

        # 随机选择 UA 和分辨率
        user_agent = random.choice(UA_LIST)
        width, height = random.choice(RESOLUTIONS)
        height = height - random.randint(0, 80)
        viewport = {"width": width, "height": height}

        # 解析代理
        effective_proxy = BROWSER_PROXY_URL
        proxy_option = parse_proxy_url(effective_proxy) if effective_proxy else None
        using_proxy = bool(proxy_option)

        logger.info(
            f"[custom-score] 开始分数测试: url={req.website_url}, "
            f"action={req.action}, enterprise={req.enterprise}"
        )

        # 启动浏览器
        playwright_instance = await async_playwright().start()
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-quic',
            '--disable-features=UseDnsHttpsSvcb',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-setuid-sandbox',
            '--no-first-run',
            '--no-zygote',
            f'--window-size={width},{height}',
            '--disable-infobars',
            '--hide-scrollbars',
            '--start-minimized',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            # CPU 优化
            '--disable-gpu',
            '--disable-gpu-compositing',
            '--disable-software-rasterizer',
            '--disable-extensions',
            '--disable-default-apps',
            '--disable-sync',
            '--disable-translate',
            '--disable-background-networking',
            '--metrics-recording-only',
            '--mute-audio',
            '--no-default-browser-check',
        ]
        if sys.platform.startswith("win"):
            browser_args.append('--window-position=-32000,-32000')

        browser = await playwright_instance.chromium.launch(
            headless=False,
            proxy=proxy_option,
            args=browser_args,
        )
        context = await browser.new_context(
            viewport=viewport,
            locale="en-US",
        )

        result = await _execute_custom_captcha_in_context(
            context=context,
            website_url=req.website_url,
            website_key=req.website_key,
            verify_url=req.verify_url,
            action=req.action,
            enterprise=req.enterprise,
            using_proxy=using_proxy,
        )

        token_elapsed_ms = int((time.time() - token_start) * 1000)

        if result is None:
            raise RuntimeError("分数测试失败：未获取到 token")

        if isinstance(result, dict):
            result["token_elapsed_ms"] = token_elapsed_ms
            # 提取指纹
            fingerprint = await _capture_page_fingerprint_from_context(context)
            result["fingerprint"] = fingerprint
            return result
        else:
            # result is token string
            return {
                "token": result,
                "token_elapsed_ms": token_elapsed_ms,
                "verify_mode": "remote_browser_page",
            }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[custom-score] 异常: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"分数测试失败: {str(e)}")
    finally:
        if acquired:
            session_manager._semaphore.release()
        try:
            if context:
                await asyncio.wait_for(context.close(), timeout=10)
        except Exception:
            pass
        try:
            if browser:
                await asyncio.wait_for(browser.close(), timeout=10)
        except Exception:
            pass
        try:
            if playwright_instance:
                await asyncio.wait_for(playwright_instance.stop(), timeout=10)
        except Exception:
            pass


async def _capture_page_fingerprint_from_context(context) -> Optional[Dict[str, Any]]:
    """从 context 的第一个页面提取指纹"""
    try:
        pages = context.pages
        if not pages:
            return None
        return await _capture_page_fingerprint(pages[0], "custom-score")
    except Exception:
        return None


async def _execute_custom_captcha_in_context(
    context,
    website_url: str,
    website_key: str,
    verify_url: str,
    action: str,
    enterprise: bool,
    using_proxy: bool,
) -> Any:
    """在真实站点执行 reCAPTCHA 并在页面内验证分数"""
    page = None
    try:
        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        primary_host = "https://www.recaptcha.net" if using_proxy else "https://www.google.com"
        secondary_host = "https://www.google.com" if primary_host == "https://www.recaptcha.net" else "https://www.recaptcha.net"
        script_path = "recaptcha/enterprise.js" if enterprise else "recaptcha/api.js"
        execute_target = "grecaptcha.enterprise.execute" if enterprise else "grecaptcha.execute"
        ready_target = "grecaptcha.enterprise.ready" if enterprise else "grecaptcha.ready"
        wait_expression = (
            "typeof grecaptcha !== 'undefined' && typeof grecaptcha.enterprise !== 'undefined' && "
            "typeof grecaptcha.enterprise.execute === 'function'"
        ) if enterprise else (
            "typeof grecaptcha !== 'undefined' && typeof grecaptcha.execute === 'function'"
        )

        logger.info(f"[custom-score] 访问真实页面: {website_url}")

        def handle_request_failed(request):
            try:
                failed_url = request.url or ""
                if not any(d in failed_url for d in ["google.com", "gstatic.com", "recaptcha.net", "antcpt.com"]):
                    return
                failure = request.failure or ""
                logger.warning(f"[custom-score] 资源加载失败: url={failed_url[:200]}, error={failure}")
            except Exception:
                pass

        page.on("requestfailed", handle_request_failed)

        # 导航到真实页面
        try:
            await page.goto(website_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            logger.warning(f"[custom-score] page.goto 失败: {type(e).__name__}: {str(e)[:200]}")
            return None

        # 等待页面完全加载
        for _ in range(20):
            try:
                ready_state = await page.evaluate("document.readyState")
                if ready_state == "complete":
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        # 模拟自然交互行为
        try:
            await page.mouse.move(320, 220)
            await page.mouse.move(520, 320, steps=12)
            await page.mouse.wheel(0, 240)
            await page.evaluate("""
                (() => {
                    try {
                        window.focus();
                        window.dispatchEvent(new Event('focus'));
                        document.dispatchEvent(new MouseEvent('mousemove', {
                            bubbles: true,
                            clientX: Math.max(32, Math.floor((window.innerWidth || 1280) * 0.4)),
                            clientY: Math.max(32, Math.floor((window.innerHeight || 720) * 0.35))
                        }));
                        window.scrollTo(0, Math.min(280, document.body?.scrollHeight || 280));
                    } catch (e) {}
                })()
            """)
        except Exception:
            pass

        # 页面预热等待
        warmup_seconds = 12
        logger.info(f"[custom-score] 真实页面预热 {warmup_seconds}s...")
        await asyncio.sleep(warmup_seconds)

        # 等待 grecaptcha 就绪
        try:
            await page.wait_for_function(wait_expression, timeout=15000)
        except Exception as e:
            logger.warning(f"[custom-score] grecaptcha 未就绪，尝试补注入: {e}")
            try:
                await page.evaluate(f"""
                    (primaryUrl, secondaryUrl) => {{
                        const existing = Array.from(document.scripts || []).some((script) => {{
                            const src = script?.src || "";
                            return src.includes('/recaptcha/');
                        }});
                        if (existing) return;
                        const urls = [primaryUrl, secondaryUrl];
                        const loadScript = (index) => {{
                            if (index >= urls.length) return;
                            const script = document.createElement('script');
                            script.src = urls[index];
                            script.async = true;
                            script.onerror = () => loadScript(index + 1);
                            document.head.appendChild(script);
                        }};
                        loadScript(0);
                    }}
                """, f"{primary_host}/{script_path}?render={website_key}", f"{secondary_host}/{script_path}?render={website_key}")
                await page.wait_for_function(wait_expression, timeout=15000)
            except Exception as inject_error:
                logger.warning(f"[custom-score] grecaptcha 最终未就绪: {inject_error}")
                return None

        # 提取浏览器指纹
        await _capture_page_fingerprint(page, "custom-score")

        # 执行 reCAPTCHA
        token = await asyncio.wait_for(
            page.evaluate(f"""
                (actionName) => {{
                    return new Promise((resolve, reject) => {{
                        const timeout = setTimeout(() => reject(new Error('timeout')), 25000);
                        try {{
                            {ready_target}(function() {{
                                {execute_target}('{website_key}', {{action: actionName}})
                                    .then(t => {{
                                        clearTimeout(timeout);
                                        resolve(t);
                                    }})
                                    .catch(e => {{
                                        clearTimeout(timeout);
                                        reject(e);
                                    }});
                            }});
                        }} catch (e) {{
                            clearTimeout(timeout);
                            reject(e);
                        }}
                    }});
                }}
            """, action),
            timeout=30,
        )

        # 额外稳定等待
        await asyncio.sleep(3)

        # 在页面中读取分数
        if verify_url:
            verify_payload = await _verify_score_in_page(page, token, verify_url)
            return {
                "token": token,
                "verify_mode": "remote_browser_page_dom",
                **verify_payload,
            }

        return token

    except Exception as e:
        logger.warning(f"[custom-score] 打码异常: {type(e).__name__}: {str(e)[:200]}")
        return None
    finally:
        if page:
            try:
                await page.close()
            except Exception:
                pass


async def _verify_score_in_page(page, token: str, verify_url: str) -> Dict[str, Any]:
    """从页面 DOM 读取分数（复用 browser_captcha.py 的逻辑）"""
    started_at = time.time()
    timeout_seconds = 25.0
    refresh_clicked = False
    last_snapshot: Dict[str, Any] = {}

    while (time.time() - started_at) < timeout_seconds:
        try:
            result = await page.evaluate("""
                () => {
                    const bodyText = ((document.body && document.body.innerText) || "")
                        .replace(/\\u00a0/g, " ")
                        .replace(/\\r/g, "");
                    const patterns = [
                        { source: "current_score", regex: /Your score is:\\s*([01](?:\\.\\d+)?)/i },
                        { source: "selected_score", regex: /Selected Score Test:[\\s\\S]{0,400}?Score:\\s*([01](?:\\.\\d+)?)/i },
                        { source: "history_score", regex: /(?:^|\\n)\\s*Score:\\s*([01](?:\\.\\d+)?)\\s*;/i },
                    ];
                    let score = null;
                    let source = "";
                    for (const item of patterns) {
                        const match = bodyText.match(item.regex);
                        if (!match) continue;
                        const parsed = Number(match[1]);
                        if (!Number.isNaN(parsed) && parsed >= 0 && parsed <= 1) {
                            score = parsed;
                            source = item.source;
                            break;
                        }
                    }
                    const uaMatch = bodyText.match(/Current User Agent:\\s*([^\\n]+)/i);
                    const ipMatch = bodyText.match(/Current IP Address:\\s*([^\\n]+)/i);
                    return {
                        score,
                        source,
                        raw_text: bodyText.slice(0, 4000),
                        current_user_agent: uaMatch ? uaMatch[1].trim() : "",
                        current_ip_address: ipMatch ? ipMatch[1].trim() : "",
                        title: document.title || "",
                        url: location.href || "",
                    };
                }
            """)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

        if isinstance(result, dict):
            last_snapshot = result
            score = result.get("score")
            if isinstance(score, (int, float)):
                elapsed_ms = int((time.time() - started_at) * 1000)
                return {
                    "verify_elapsed_ms": elapsed_ms,
                    "verify_http_status": None,
                    "verify_result": {
                        "success": True,
                        "score": score,
                        "source": result.get("source") or "antcpt_dom",
                        "raw_text": result.get("raw_text") or "",
                        "current_user_agent": result.get("current_user_agent") or "",
                        "current_ip_address": result.get("current_ip_address") or "",
                        "page_title": result.get("title") or "",
                        "page_url": result.get("url") or "",
                    },
                }

        # 尝试点击刷新按钮
        if not refresh_clicked and (time.time() - started_at) >= 2:
            refresh_clicked = True
            try:
                await page.evaluate("""
                    () => {
                        const nodes = Array.from(
                            document.querySelectorAll('button, input[type="button"], input[type="submit"], a')
                        );
                        const target = nodes.find((node) => {
                            const text = (node.innerText || node.textContent || node.value || "").trim();
                            return /Refresh score now!?/i.test(text);
                        });
                        if (target) {
                            target.click();
                            return true;
                        }
                        return false;
                    }
                """)
            except Exception:
                pass

        await asyncio.sleep(0.5)

    elapsed_ms = int((time.time() - started_at) * 1000)
    if not isinstance(last_snapshot, dict):
        last_snapshot = {"raw": last_snapshot}

    return {
        "verify_elapsed_ms": elapsed_ms,
        "verify_http_status": None,
        "verify_result": {
            "success": False,
            "score": None,
            "source": "antcpt_dom_timeout",
            "raw_text": last_snapshot.get("raw_text") or "",
            "current_user_agent": last_snapshot.get("current_user_agent") or "",
            "current_ip_address": last_snapshot.get("current_ip_address") or "",
            "page_title": last_snapshot.get("title") or "",
            "page_url": last_snapshot.get("url") or "",
            "error": last_snapshot.get("error") or "未在页面中读取到分数",
        },
    }


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
    )
