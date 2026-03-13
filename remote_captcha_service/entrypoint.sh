#!/bin/bash
set -e

# 启动 Xvfb 虚拟显示器
echo "🖥️ 启动 Xvfb 虚拟显示器..."
Xvfb :99 -screen 0 ${XVFB_WHD:-1920x1080x24} -ac +extension GLX +render -noreset &
export DISPLAY=:99

# 等待 Xvfb 就绪
sleep 1

# 启动窗口管理器
fluxbox &

echo "🚀 启动远程有头打码服务..."
exec python server.py
