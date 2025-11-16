#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于akshare的A股数据收集器
支持获取行情数据、财务数据、估值数据、因子数据、宏观数据和舆情数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

class AkshareStockCollector:
    def __init__(self):
        self.stocks = {
            '格尔软件': '603232',
            '歌尔股份': '002241',
            '中国长城': '000066',
            '科大讯飞': '002230',
            '片仔癀': '600436'
        }
        self.data_file = 'akshare_stock_data.xlsx'

    def get_price_data(self, start_date=None, end_date=None):
        """获取行情数据

        Parameters:
        start_date: 开始日期，格式为'YYYYMMDD'，默认一年前
        end_date: 结束日期，格式为'YYYYMMDD'，默认今天
        """
        print("正在获取行情数据...")
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        price_data = {}
        for name, code in self.stocks.items():
            try:
                # 获取历史行情数据
                df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                      start_date=start_date, end_date=end_date,
                                      adjust="qfq")

                # 计算技术指标
                df = self.calculate_technical_indicators(df)
                price_data[name] = df
                print(f"  ✓ 已获取 {name} 行情数据 ({start_date} 到 {end_date}, {len(df)} 条记录)")

            except Exception as e:
                print(f"  ✗ 获取 {name} 行情数据失败: {e}")
                price_data[name] = None

        return price_data

    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        # 移动平均线
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        df['MA60'] = df['收盘'].rolling(60).mean()

        # MACD
        exp1 = df['收盘'].ewm(span=12).mean()
        exp2 = df['收盘'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林带
        df['BB_Middle'] = df['收盘'].rolling(20).mean()
        bb_std = df['收盘'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std

        return df

    def get_financial_data(self):
        """获取财务数据 - 使用正确的akshare API"""
        print("正在获取财务数据...")
        financial_data = {}

        for name, code in self.stocks.items():
            try:
                # 获取最新财务报表数据
                # 资产负债表
                balance_sheet = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")

                # 利润表
                income_statement = ak.stock_financial_report_sina(stock=code, symbol="利润表")

                # 现金流量表
                cash_flow = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")

                financial_data[name] = {
                    'balance_sheet': balance_sheet,
                    'income_statement': income_statement,
                    'cash_flow': cash_flow
                }
                print(f"  ✓ 已获取 {name} 财务数据")

            except Exception as e:
                print(f"  ✗ 获取 {name} 财务数据失败: {e}")
                financial_data[name] = None

        return financial_data

    def get_valuation_data(self):
        """获取估值数据 - 暂时跳过，后续实现"""
        print("正在获取估值数据...")
        print("  ⚠  估值数据接口暂不可用，跳过获取")
        return {}

    def get_factor_data(self):
        """获取因子数据 - 暂时跳过，后续实现"""
        print("正在获取因子数据...")
        print("  ⚠  因子数据接口暂不可用，跳过获取")
        return {}

    def get_macro_data(self):
        """获取宏观数据"""
        print("正在获取宏观数据...")
        macro_data = {}

        try:
            # 获取CPI数据
            cpi_data = ak.macro_china_cpi()
            macro_data['cpi'] = cpi_data

            # 获取PPI数据
            ppi_data = ak.macro_china_ppi()
            macro_data['ppi'] = ppi_data

            # 获取GDP数据
            gdp_data = ak.macro_china_gdp()
            macro_data['gdp'] = gdp_data

            print(f"  ✓ 已获取宏观数据")

        except Exception as e:
            print(f"  ✗ 获取宏观数据失败: {e}")
            macro_data = None

        return macro_data

    def get_sentiment_data(self):
        """获取舆情数据"""
        print("正在获取舆情数据...")
        sentiment_data = {}

        for name, code in self.stocks.items():
            try:
                # 获取新闻舆情数据
                news = ak.stock_news_em(symbol=code)
                sentiment_data[name] = news
                print(f"  ✓ 已获取 {name} 舆情数据 ({len(news)} 条记录)")

            except Exception as e:
                print(f"  ✗ 获取 {name} 舆情数据失败: {e}")
                sentiment_data[name] = None

        return sentiment_data

    def collect_all_data(self, start_date=None, end_date=None):
        """收集所有数据

        Parameters:
        start_date: 开始日期，格式为'YYYYMMDD'
        end_date: 结束日期，格式为'YYYYMMDD'
        """
        print("开始收集所有股票数据...")

        all_data = {
            'price_data': self.get_price_data(start_date, end_date),
            'financial_data': self.get_financial_data(),
            'valuation_data': self.get_valuation_data(),
            'factor_data': self.get_factor_data(),
            'macro_data': self.get_macro_data(),
            'sentiment_data': self.get_sentiment_data(),
            'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print("数据收集完成!")
        return all_data

    def save_data(self, data):
        """保存数据到Excel文件"""
        try:
            with pd.ExcelWriter(self.data_file, engine='openpyxl') as writer:
                # 保存价格数据
                price_data = data.get('price_data', {})
                for stock_name, df in price_data.items():
                    if df is not None:
                        df.to_excel(writer, sheet_name=f'{stock_name}_价格')

                # 保存财务数据
                financial_data = data.get('financial_data', {})
                for stock_name, financial_dict in financial_data.items():
                    if financial_dict is not None:
                        for table_name, table_data in financial_dict.items():
                            if table_data is not None:
                                table_data.to_excel(writer, sheet_name=f'{stock_name}_{table_name}')

                # 保存估值数据
                valuation_data = data.get('valuation_data', {})
                for stock_name, df in valuation_data.items():
                    if df is not None:
                        df.to_excel(writer, sheet_name=f'{stock_name}_估值')

                # 保存因子数据
                factor_data = data.get('factor_data', {})
                for factor_name, df in factor_data.items():
                    if df is not None:
                        df.to_excel(writer, sheet_name=f'因子_{factor_name}')

                # 保存宏观数据
                macro_data = data.get('macro_data', {})
                for macro_name, df in macro_data.items():
                    if df is not None:
                        df.to_excel(writer, sheet_name=f'宏观_{macro_name}')

                # 保存舆情数据
                sentiment_data = data.get('sentiment_data', {})
                for stock_name, df in sentiment_data.items():
                    if df is not None:
                        df.to_excel(writer, sheet_name=f'{stock_name}_舆情')

                # 保存元数据
                meta_data = {'collection_time': [data.get('collection_time', '')]}
                pd.DataFrame(meta_data).to_excel(writer, sheet_name='元数据')

            print(f"数据已保存到Excel文件: {self.data_file}")
            return True

        except Exception as e:
            print(f"保存数据到Excel失败: {e}")
            return False

    def get_data_summary(self, data):
        """获取数据摘要"""
        if not data:
            print("无数据可用")
            return

        print("\n" + "="*50)
        print("数据摘要")
        print("="*50)
        print(f"收集时间: {data.get('collection_time', '未知')}")

        for data_type, data_dict in data.items():
            if data_type == 'collection_time':
                continue

            if isinstance(data_dict, dict):
                valid_count = sum(1 for v in data_dict.values() if v is not None)
                total_count = len(data_dict)
                print(f"{data_type}: {valid_count}/{total_count} 只股票数据有效")
            else:
                if data_dict is not None:
                    print(f"{data_type}: 数据有效")
                else:
                    print(f"{data_type}: 无数据")

def main():
    """
    主函数 - 支持自定义时间范围的数据收集
    用法:
    python akshare_stock_collector.py                          # 默认获取最近一年数据
    python akshare_stock_collector.py --start 20240101        # 从2024-01-01到今天
    python akshare_stock_collector.py --end 20241231          # 从最早到2024-12-31
    python akshare_stock_collector.py --start 20240101 --end 20241231  # 指定时间范围
    """
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='A股数据收集器 (基于akshare)')
    parser.add_argument('--start', type=str, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYYMMDD)')

    args = parser.parse_args()

    # 创建数据收集器
    collector = AkshareStockCollector()

    # 收集所有数据（支持自定义时间范围）
    data = collector.collect_all_data(args.start, args.end)

    # 保存数据
    if collector.save_data(data):
        # 显示数据摘要
        collector.get_data_summary(data)
        print(f"\n数据已保存到: {collector.data_file}")
    else:
        print("数据保存失败")

if __name__ == "__main__":
    main()