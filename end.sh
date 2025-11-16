#!/bin/bash

# 股票预测系统清理脚本
# 作者: Zulu AI助手
# 日期: 2025-11-15

echo "🧹 清理股票预测系统..."

# 设置工作目录
cd "$(dirname "$0")"

# 查找并杀死streamlit进程
echo "🔫 停止streamlit服务..."
pkill -f "streamlit run streamlit_app.py" 2>/dev/null
pkill -f "python.*streamlit" 2>/dev/null

# 检查是否成功停止
if pgrep -f "streamlit" > /dev/null; then
    echo "❌ 无法停止streamlit进程，尝试强制停止..."
    pkill -9 -f "streamlit" 2>/dev/null
fi

# 清理缓存文件
echo "🗑️  清理缓存文件..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".streamlit" -type d -exec rm -rf {} + 2>/dev/null

# 清理可能生成的临时文件
rm -f .streamlit_cache/* 2>/dev/null
rm -rf .streamlit/ 2>/dev/null

# 清理MacOS系统文件
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "._*" -delete 2>/dev/null

# 清理日志文件（如果有）
rm -f streamlit.log 2>/dev/null
rm -f *.log 2>/dev/null

echo "✅ 清理完成！"
echo ""
echo "📋 已清理的项目:"
echo "   - streamlit进程"
echo "   - Python缓存文件 (*.pyc)"
echo "   - __pycache__ 目录"
echo "   - .streamlit 配置目录"
echo "   - 系统临时文件 (.DS_Store, ._*)"
echo "   - 日志文件"

# 检查是否还有残留进程
if pgrep -f "streamlit" > /dev/null; then
    echo "⚠️  警告：仍有streamlit进程在运行，请手动检查:"
    pgrep -fl "streamlit"
else
    echo "✅ 所有streamlit进程已成功停止"
fi