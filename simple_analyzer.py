#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版股票数据分析器
使用纯文本分析，不依赖图形库
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleStockAnalyzer:
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
                'sentiment_data': {}
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

        print("\n" + "="*60)
        print("价格趋势分析")
        print("="*60)

        # 计算收益率统计
        returns_stats = []
        for stock, df in self.data['price_data'].items():
            if df is not None and not df.empty:
                # 计算技术指标
                current_price = df['收盘'].iloc[-1]
                ma5 = df['MA5'].iloc[-1]
                ma20 = df['MA20'].iloc[-1]
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None

                # 计算收益率
                returns = df['收盘'].pct_change().dropna()

                stats = {
                    '股票': stock,
                    '当前价格': f"{current_price:.2f}",
                    '5日均价': f"{ma5:.2f}",
                    '20日均价': f"{ma20:.2f}",
                    'RSI': f"{rsi:.1f}" if rsi is not None else "N/A",
                    '趋势': "上涨" if current_price > ma5 > ma20 else
                           "下跌" if current_price < ma5 < ma20 else "震荡",
                    '平均日收益': f"{returns.mean():.3%}",
                    '波动率': f"{returns.std():.3%}",
                    '夏普比率': f"{returns.mean() / returns.std() * np.sqrt(252):.2f}" if returns.std() > 0 else "N/A"
                }
                returns_stats.append(stats)

        # 输出表格
        if returns_stats:
            df_stats = pd.DataFrame(returns_stats)
            print(df_stats.to_string(index=False, justify='center'))

    def analyze_financial_ratios(self):
        """分析财务比率"""
        if not self.data or 'financial_data' not in self.data:
            print("无财务数据可用")
            return

        print("\n" + "="*60)
        print("财务比率分析")
        print("="*60)

        financial_ratios = []

        for stock, financial_dict in self.data['financial_data'].items():
            if financial_dict and '利润表' in financial_dict and '资产负债表' in financial_dict:
                income_stmt = financial_dict['利润表']
                balance_sheet = financial_dict['资产负债表']

                # 计算关键财务比率
                try:
                    # 获取最新财务数据
                    if not income_stmt.empty and not balance_sheet.empty:
                        # 尝试不同的列名
                        revenue_col = next((col for col in ['营业总收入', '营业收入', '收入'] if col in income_stmt.columns), None)
                        profit_col = next((col for col in ['净利润', '净收益', '利润'] if col in income_stmt.columns), None)
                        assets_col = next((col for col in ['资产总计', '总资产', '资产'] if col in balance_sheet.columns), None)
                        equity_col = next((col for col in ['股东权益合计', '净资产', '所有者权益'] if col in balance_sheet.columns), None)

                        if revenue_col and profit_col and assets_col and equity_col:
                            latest_revenue = income_stmt[revenue_col].iloc[0]
                            latest_profit = income_stmt[profit_col].iloc[0]
                            latest_assets = balance_sheet[assets_col].iloc[0]
                            latest_equity = balance_sheet[equity_col].iloc[0]

                            ratios = {
                                '股票': stock,
                                '营业收入(亿)': f"{latest_revenue / 1e8:.2f}",
                                '净利润(亿)': f"{latest_profit / 1e8:.2f}",
                                '净利润率': f"{latest_profit / latest_revenue:.2%}" if latest_revenue > 0 else "N/A",
                                '净资产收益率': f"{latest_profit / latest_equity:.2%}" if latest_equity > 0 else "N/A",
                                '资产收益率': f"{latest_profit / latest_assets:.2%}" if latest_assets > 0 else "N/A"
                            }
                            financial_ratios.append(ratios)

                except Exception as e:
                    print(f"计算 {stock} 财务比率时出错: {e}")

        if financial_ratios:
            ratios_df = pd.DataFrame(financial_ratios)
            print(ratios_df.to_string(index=False, justify='center'))

    def analyze_sentiment(self):
        """分析舆情数据"""
        if not self.data or 'sentiment_data' not in self.data:
            print("无舆情数据可用")
            return

        print("\n" + "="*60)
        print("舆情分析")
        print("="*60)

        for stock, df in self.data['sentiment_data'].items():
            if df is not None and not df.empty:
                print(f"\n{stock}:")
                print(f"  新闻数量: {len(df)}")

                if '发布时间' in df.columns:
                    latest_news = df['发布时间'].max()
                    print(f"  最新新闻: {latest_news}")

                if '新闻标题' in df.columns:
                    recent_titles = df['新闻标题'].head(3).tolist()
                    print("  近期新闻标题:")
                    for i, title in enumerate(recent_titles, 1):
                        print(f"    {i}. {title[:50]}...")

    def generate_investment_recommendation(self):
        """生成投资建议"""
        print("\n" + "="*60)
        print("投资建议")
        print("="*60)

        if not self.data:
            print("无数据可用")
            return

        recommendations = []

        # 分析价格数据
        if 'price_data' in self.data:
            for stock, df in self.data['price_data'].items():
                if df is not None and not df.empty:
                    current_price = df['收盘'].iloc[-1]
                    ma20 = df['MA20'].iloc[-1]
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50

                    # 简单技术分析
                    if current_price > ma20 and rsi < 70:
                        signal = "买入"
                        reason = "价格在20日均线上方且RSI未超买"
                    elif current_price < ma20 and rsi > 30:
                        signal = "持有"
                        reason = "价格在20日均线下方但RSI未超卖"
                    else:
                        signal = "观望"
                        reason = "等待更明确信号"

                    recommendations.append({
                        '股票': stock,
                        '建议': signal,
                        '理由': reason,
                        '当前价': f"{current_price:.2f}",
                        '20日均价': f"{ma20:.2f}"
                    })

        # 输出建议
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            print(rec_df.to_string(index=False, justify='center'))

    def generate_full_report(self):
        """生成完整分析报告"""
        print("正在生成股票分析报告...")
        print("="*60)

        # 加载数据
        if not self.load_data():
            return

        # 执行各项分析
        self.analyze_price_trends()
        self.analyze_financial_ratios()
        self.analyze_sentiment()
        self.generate_investment_recommendation()

        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
        print("建议下一步操作:")
        print("1. 定期运行数据收集器更新数据")
        print("2. 关注财务指标优秀的股票")
        print("3. 结合技术分析和基本面分析制定策略")
        print("4. 设置价格预警点位")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='简化版股票数据分析器')
    parser.add_argument('--file', type=str, default='akshare_stock_data.xlsx',
                       help='数据文件路径')
    parser.add_argument('--report', action='store_true', help='生成完整报告')

    args = parser.parse_args()

    analyzer = SimpleStockAnalyzer(args.file)

    if args.report:
        analyzer.generate_full_report()
    else:
        if analyzer.load_data():
            analyzer.analyze_price_trends()

if __name__ == "__main__":
    main()