#!/bin/bash

# 股票预测系统启动脚本
# 作者: Zulu AI助手
# 日期: 2025-11-15

echo "🚀 启动股票预测系统..."

# 设置工作目录
cd "$(dirname "$0")"

# 检查Python依赖
echo "📦 检查Python依赖..."
if ! python3 -c "
import sys
try:
    import streamlit
    import pandas
    import akshare
    import plotly
    import sklearn
    import tensorflow
    print('✅ 所有依赖检查通过')
    sys.exit(0)
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  检查依赖时出错: {e}')
    sys.exit(1)
" 2>/dev/null; then
    echo "❌ 缺少必要的Python依赖，请先运行: pip3 install -r requirements.txt"
    exit 1
else
    echo "✅ 所有Python依赖已安装"
fi

# 添加Python包路径到PATH（如果需要）
export PATH="/Users/luoliming/Library/Python/3.9/bin:$PATH"

# 设置环境变量
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 检查streamlit是否在PATH中
if ! command -v streamlit &> /dev/null; then
    # 如果不在PATH中，使用完整路径
    STREAMLIT_PATH="/Users/luoliming/Library/Python/3.9/bin/streamlit"
    if [ -f "$STREAMLIT_PATH" ]; then
        echo "🔧 使用streamlit完整路径: $STREAMLIT_PATH"
        $STREAMLIT_PATH run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    else
        echo "❌ 找不到streamlit，请检查安装"
        exit 1
    fi
else
    echo "✅ streamlit已在PATH中找到"
    streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
fi