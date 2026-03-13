FROM mcr.microsoft.com/playwright/python:v1.49.1-noble

WORKDIR /app

# 安装 Xvfb 和窗口管理器
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    fluxbox \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Playwright 浏览器
RUN python -m playwright install chromium

# 复制应用代码
COPY server.py .

# 启动脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8060

ENTRYPOINT ["/entrypoint.sh"]
