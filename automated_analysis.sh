#!/bin/bash

# 股票数据分析自动化脚本
# 每日自动更新数据并生成分析报告

# 设置工作目录
cd "$(dirname "$0")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 创建备份目录
create_backup() {
    local backup_dir="backup/$(date '+%Y%m%d')"
    mkdir -p "$backup_dir"

    if [ -f "akshare_stock_data.xlsx" ]; then
        cp "akshare_stock_data.xlsx" "$backup_dir/"
        log_info "数据文件已备份到 $backup_dir/"
    fi
}

# 更新数据
update_data() {
    log_info "开始更新股票数据..."

    # 计算日期范围（最近30天）
    end_date=$(date '+%Y%m%d')
    start_date=$(date -v-30d '+%Y%m%d')

    python3 akshare_stock_collector.py --start "$start_date" --end "$end_date"

    if [ $? -eq 0 ]; then
        log_success "股票数据更新完成"
    else
        log_error "股票数据更新失败"
        return 1
    fi
}

# 生成分析报告
generate_report() {
    log_info "开始生成分析报告..."

    python3 simple_analyzer.py --report > "analysis_report_$(date '+%Y%m%d_%H%M%S').txt"

    if [ $? -eq 0 ]; then
        log_success "分析报告生成完成"
        # 显示报告摘要
        echo ""
        echo "=== 最新分析报告摘要 ==="
        tail -20 "analysis_report_$(date '+%Y%m%d_%H%M%S').txt" | head -15
    else
        log_error "分析报告生成失败"
        return 1
    fi
}

# 清理旧文件
cleanup_old_files() {
    log_info "清理旧文件..."

    # 保留最近7天的报告
    find . -name "analysis_report_*.txt" -mtime +7 -delete
    find . -name "backup" -type d -mtime +30 -exec rm -rf {} + 2>/dev/null

    log_success "旧文件清理完成"
}

# 主函数
main() {
    echo "=========================================="
    echo "   股票数据分析自动化脚本"
    echo "   开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # 创建备份
    create_backup

    # 更新数据
    if update_data; then
        # 生成报告
        generate_report
    else
        log_error "数据更新失败，跳过报告生成"
    fi

    # 清理旧文件
    cleanup_old_files

    echo "=========================================="
    echo "   脚本执行完成: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
}

# 执行主函数
main "$@"