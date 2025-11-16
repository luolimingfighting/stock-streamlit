#!/bin/bash

# 股票预测系统启动脚本（简化版）

echo "Starting Stock Prediction System..."

# 设置工作目录
cd "$(dirname "$0")"

# 检查主要依赖
echo "Checking Python dependencies..."
if ! python3 -c "import streamlit, pandas, akshare, plotly" 2>/dev/null; then
    echo "Error: Missing Python dependencies. Please run: pip3 install -r requirements.txt"
    exit 1
fi

echo "All dependencies are installed."

# 设置PATH
export PATH="/Users/luoliming/Library/Python/3.9/bin:$PATH"

# 设置Streamlit环境变量来禁用部署提示
export STREAMLIT_GLOBAL_DISABLE_DEPLOY_WARNING=1
export STREAMLIT_SERVER_ENABLE_STATIC_SERVE=1
export STREAMLIT_GLOBAL_DISABLE_WATCHDOG_WARNING=1

# 启动应用
echo "Starting Streamlit application..."
/Users/luoliming/Library/Python/3.9/bin/streamlit run streamlit_app.py \
    --server.port=8501 \
    --global.suppressDeprecationWarnings=true \
    --logger.level=error