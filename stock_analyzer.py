#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票数据分析器
支持对akshare和聚宽收集的数据进行分析和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StockAnalyzer:
    def __init__(self, data_file='akshare_stock_data.xlsx'):
        self.data_file = data_file
        self.data = None
        self.stocks = ['格尔软件', '歌尔股份', '中国长城', '科大讯飞', '片仔癀']

    def load_data(self):
        """从Excel文件加载数据"""
        try:
            print(f"正在加载数据文件: {self.data_file}")
            excel_file = pd.ExcelFile(self.data_file)

            data = {
                'price_data': {},
                'financial_data': {},
                'sentiment_data': {},
                'macro_data': {}
            }

            # 读取价格数据
            for stock in self.stocks:
                sheet_name = f'{stock}_价格'
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                    data['price_data'][stock] = df

            # 读取财务数据
            for stock in self.stocks:
                for table in ['资产负债表', '利润表', '现金流量表']:
                    sheet_name = f'{stock}_{table}'
                    if sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                        if stock not in data['financial_data']:
                            data['financial_data'][stock] = {}
                        data['financial_data'][stock][table] = df

            # 读取舆情数据
            for stock in self.stocks:
                sheet_name = f'{stock}_舆情'
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    data['sentiment_data'][stock] = df

            # 读取宏观数据
            for macro in ['cpi', 'ppi', 'gdp']:
                sheet_name = f'宏观_{macro}'
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    data['macro_data'][macro] = df

            self.data = data
            print("数据加载完成!")
            return True

        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def analyze_price_trends(self):
        """分析价格趋势"""
        if not self.data or 'price_data' not in self.data:
            print("无价格数据可用")
            return

        print("\n" + "="*50)
        print("价格趋势分析")
        print("="*50)

        # 创建价格趋势图
        plt.figure(figsize=(15, 10))

        for i, (stock, df) in enumerate(self.data['price_data'].items(), 1):
            if df is not None and not df.empty:
                plt.subplot(3, 2, i)
                plt.plot(df.index, df['收盘'], label=stock, linewidth=2)
                plt.plot(df.index, df['MA5'], label='MA5', linestyle='--', alpha=0.7)
                plt.plot(df.index, df['MA20'], label='MA20', linestyle='--', alpha=0.7)
                plt.title(f'{stock} 价格趋势')
                plt.xlabel('日期')
                plt.ylabel('价格')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('price_trends.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 计算收益率统计
        print("\n收益率统计:")
        returns_stats = []
        for stock, df in self.data['price_data'].items():
            if df is not None and not df.empty:
                returns = df['收盘'].pct_change().dropna()
                stats = {
                    '股票': stock,
                    '平均日收益率': f"{returns.mean():.4%}",
                    '收益率标准差': f"{returns.std():.4%}",
                    '夏普比率': f"{returns.mean() / returns.std() * np.sqrt(252):.2f}",
                    '最大回撤': f"{((df['收盘'] / df['收盘'].cummax()) - 1).min():.2%}"
                }
                returns_stats.append(stats)

        returns_df = pd.DataFrame(returns_stats)
        print(returns_df.to_string(index=False))

    def analyze_technical_indicators(self):
        """分析技术指标"""
        if not self.data or 'price_data' not in self.data:
            print("无价格数据可用")
            return

        print("\n" + "="*50)
        print("技术指标分析")
        print("="*50)

        # 创建技术指标图
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, (stock, df) in enumerate(self.data['price_data'].items()):
            if df is not None and not df.empty and i < len(axes):
                ax = axes[i]

                # MACD
                ax.plot(df.index, df['MACD'], label='MACD', linewidth=1.5)
                ax.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1.5)
                ax.bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.3)
                ax.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'{stock} MACD')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('technical_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_financial_ratios(self):
        """分析财务比率"""
        if not self.data or 'financial_data' not in self.data:
            print("无财务数据可用")
            return

        print("\n" + "="*50)
        print("财务比率分析")
        print("="*50)

        financial_ratios = []

        for stock, financial_dict in self.data['financial_data'].items():
            if financial_dict and '利润表' in financial_dict and '资产负债表' in financial_dict:
                income_stmt = financial_dict['利润表']
                balance_sheet = financial_dict['资产负债表']

                # 计算关键财务比率
                try:
                    # 获取最新财务数据
                    latest_income = income_stmt.iloc[0] if not income_stmt.empty else None
                    latest_balance = balance_sheet.iloc[0] if not balance_sheet.empty else None

                    if latest_income is not None and latest_balance is not None:
                        ratios = {
                            '股票': stock,
                            '营业收入(亿)': f"{latest_income.get('营业总收入', 0) / 1e8:.2f}",
                            '净利润(亿)': f"{latest_income.get('净利润', 0) / 1e8:.2f}",
                            '毛利率': f"{latest_income.get('毛利率', 0):.2%}",
                            '净资产收益率': f"{latest_income.get('净资产收益率', 0):.2%}",
                            '资产负债率': f"{latest_balance.get('资产负债率', 0):.2%}"
                        }
                        financial_ratios.append(ratios)

                except Exception as e:
                    print(f"计算 {stock} 财务比率时出错: {e}")

        if financial_ratios:
            ratios_df = pd.DataFrame(financial_ratios)
            print(ratios_df.to_string(index=False))

            # 创建财务比率可视化
            plt.figure(figsize=(12, 8))

            # 净资产收益率比较
            roe_data = [(r['股票'], float(r['净资产收益率'].strip('%')))
                       for r in financial_ratios if '净资产收益率' in r]
            stocks, roe_values = zip(*roe_data)

            plt.subplot(2, 2, 1)
            plt.bar(stocks, roe_values)
            plt.title('净资产收益率(ROE)比较')
            plt.xticks(rotation=45)
            plt.ylabel('ROE (%)')

            # 毛利率比较
            gross_margin_data = [(r['股票'], float(r['毛利率'].strip('%')))
                               for r in financial_ratios if '毛利率' in r]
            if gross_margin_data:
                stocks, gm_values = zip(*gross_margin_data)
                plt.subplot(2, 2, 2)
                plt.bar(stocks, gm_values)
                plt.title('毛利率比较')
                plt.xticks(rotation=45)
                plt.ylabel('毛利率 (%)')

            plt.tight_layout()
            plt.savefig('financial_ratios.png', dpi=300, bbox_inches='tight')
            plt.show()

    def analyze_sentiment(self):
        """分析舆情数据"""
        if not self.data or 'sentiment_data' not in self.data:
            print("无舆情数据可用")
            return

        print("\n" + "="*50)
        print("舆情分析")
        print("="*50)

        sentiment_stats = []

        for stock, df in self.data['sentiment_data'].items():
            if df is not None and not df.empty:
                stats = {
                    '股票': stock,
                    '新闻数量': len(df),
                    '最新新闻时间': df['发布时间'].max() if '发布时间' in df.columns else '未知',
                    '新闻来源分布': df['文章来源'].value_counts().to_dict() if '文章来源' in df.columns else {}
                }
                sentiment_stats.append(stats)

        # 输出舆情统计
        for stats in sentiment_stats:
            print(f"\n{stats['股票']}:")
            print(f"  新闻数量: {stats['新闻数量']}")
            print(f"  最新新闻时间: {stats['最新新闻时间']}")
            if stats['新闻来源分布']:
                print("  新闻来源分布:")
                for source, count in list(stats['新闻来源分布'].items())[:3]:
                    print(f"    {source}: {count}")

    def generate_report(self):
        """生成综合分析报告"""
        print("正在生成股票分析报告...")

        # 加载数据
        if not self.load_data():
            return

        # 执行各项分析
        self.analyze_price_trends()
        self.analyze_technical_indicators()
        self.analyze_financial_ratios()
        self.analyze_sentiment()

        print("\n" + "="*50)
        print("分析完成！已生成以下图表:")
        print("="*50)
        print("1. price_trends.png - 价格趋势图")
        print("2. technical_indicators.png - 技术指标图")
        print("3. financial_ratios.png - 财务比率图")
        print("\n建议下一步:")
        print("- 定期运行数据收集器更新数据")
        print("- 基于分析结果制定投资策略")
        print("- 设置价格预警和监控机制")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='股票数据分析器')
    parser.add_argument('--file', type=str, default='akshare_stock_data.xlsx',
                       help='数据文件路径 (默认: akshare_stock_data.xlsx)')
    parser.add_argument('--analysis', choices=['price', 'technical', 'financial', 'sentiment', 'all'],
                       default='all', help='分析类型')

    args = parser.parse_args()

    analyzer = StockAnalyzer(args.file)

    if not analyzer.load_data():
        return

    if args.analysis == 'price' or args.analysis == 'all':
        analyzer.analyze_price_trends()

    if args.analysis == 'technical' or args.analysis == 'all':
        analyzer.analyze_technical_indicators()

    if args.analysis == 'financial' or args.analysis == 'all':
        analyzer.analyze_financial_ratios()

    if args.analysis == 'sentiment' or args.analysis == 'all':
        analyzer.analyze_sentiment()

if __name__ == "__main__":
    main()