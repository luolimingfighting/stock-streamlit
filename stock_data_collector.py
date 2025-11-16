from jqdatasdk import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

class StockDataCollector:
    def __init__(self):
        self.stocks = {
            '格尔软件': '603232.XSHG',
            '歌尔股份': '002241.XSHE', 
            '中国长城': '000066.XSHE',
            '科大讯飞': '002230.XSHE',
            '片仔癀': '600436.XSHG'
        }
        self.data_file = 'stock_data.xlsx'

    def auth_jqdata(self):
        """认证聚宽账号"""
        try:
            auth('18810062836', 'Llm666666')  # 请替换为您的聚宽账号
            print(f"认证成功，剩余调用次数: {get_query_count()}")
            return True
        except Exception as e:
            print(f"认证失败: {e}")
            return False

    def get_trading_days_range(self, start_date=None, end_date=None):
        """获取指定时间范围内的交易日"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # 默认获取最近100个交易日
            start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')

        trade_days = get_trade_days(start_date=start_date, end_date=end_date)
        return trade_days[0], trade_days[-1] if len(trade_days) > 0 else (None, None)

    def get_price_data(self, start_date=None, end_date=None):
        """获取行情数据（包含技术指标）

        Parameters:
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'
        """
        print("正在获取行情数据...")
        start_date, end_date = self.get_trading_days_range(start_date, end_date)

        price_data = {}
        for name, code in self.stocks.items():
            try:
                # 获取基础行情数据
                df = get_price(code, start_date=start_date, end_date=end_date,
                             frequency='daily', fields=['open', 'close', 'high', 'low',
                                                      'volume', 'money', 'pre_close'])

                # 计算技术指标
                df = self.calculate_technical_indicators(df)
                price_data[name] = df
                print(f"  ✓ 已获取 {name} 行情数据 ({start_date} 到 {end_date})")

            except Exception as e:
                print(f"  ✗ 获取 {name} 行情数据失败: {e}")
                price_data[name] = None

        return price_data

    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        # 移动平均线
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林带
        df['BB_Middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std

        return df

    def get_financial_data(self):
        """获取财务数据"""
        print("正在获取财务数据...")
        financial_data = {}

        for name, code in self.stocks.items():
            try:
                # 获取资产负债表
                balance_sheet = get_fundamentals(query(
                    balance
                ).filter(
                    balance.code == code
                ), date=datetime.now().strftime('%Y-%m-%d'))

                financial_data[name] = {
                    'balance_sheet': balance_sheet
                }
                print(f"  ✓ 已获取 {name} 财务数据")

            except Exception as e:
                print(f"  ✗ 获取 {name} 财务数据失败: {e}")
                financial_data[name] = None

        return financial_data

    def get_valuation_data(self, start_date=None, end_date=None):
        """获取估值指标

        Parameters:
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'
        """
        print("正在获取估值数据...")
        start_date, end_date = self.get_trading_days_range(start_date, end_date)

        valuation_data = {}
        for name, code in self.stocks.items():
            try:
                # 使用 get_fundamentals 获取估值数据
                q = query(valuation).filter(valuation.code == code)
                df = get_fundamentals(q, date=end_date)

                if df is not None and not df.empty:
                    valuation_data[name] = df
                    print(f"  ✓ 已获取 {name} 估值数据 (日期: {end_date})")
                else:
                    print(f"  ⚠  {name} 估值数据为空")
                    valuation_data[name] = None

            except Exception as e:
                print(f"  ✗ 获取 {name} 估值数据失败: {e}")
                valuation_data[name] = None

        return valuation_data

    def collect_all_data(self, start_date=None, end_date=None):
        """收集所有数据

        Parameters:
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'
        """
        print("开始收集所有股票数据...")

        all_data = {
            'price_data': self.get_price_data(start_date, end_date),
            'financial_data': self.get_financial_data(),
            'valuation_data': self.get_valuation_data(start_date, end_date),
            'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print("数据收集完成!")
        return all_data

    def check_data_freshness(self, data):
        """检查数据新鲜度"""
        if not data or 'collection_time' not in data:
            return False

        collection_time = datetime.strptime(data['collection_time'], '%Y-%m-%d %H:%M:%S')
        current_time = datetime.now()

        # 如果数据是今天收集的，则认为是最新的
        return collection_time.date() == current_time.date()

    def save_data(self, data):
        """保存数据到Excel文件"""
        try:
            # 创建Excel写入器
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

                # 保存元数据
                meta_data = {'collection_time': [data.get('collection_time', '')]}
                pd.DataFrame(meta_data).to_excel(writer, sheet_name='元数据')

            print(f"数据已保存到Excel文件: {self.data_file}")
            return True
        except Exception as e:
            print(f"保存数据到Excel失败: {e}")
            return False

    def load_data(self):
        """从Excel文件加载数据"""
        try:
            if os.path.exists(self.data_file):
                data = {
                    'price_data': {},
                    'financial_data': {},
                    'valuation_data': {},
                    'collection_time': ''
                }

                # 读取Excel文件
                excel_file = pd.ExcelFile(self.data_file)

                # 读取元数据
                if '元数据' in excel_file.sheet_names:
                    meta_df = pd.read_excel(excel_file, sheet_name='元数据')
                    if not meta_df.empty:
                        data['collection_time'] = meta_df['collection_time'].iloc[0] if 'collection_time' in meta_df.columns else ''

                # 读取各工作表数据
                for sheet_name in excel_file.sheet_names:
                    if sheet_name == '元数据':
                        continue

                    if sheet_name.endswith('_价格'):
                        stock_name = sheet_name.replace('_价格', '')
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                        data['price_data'][stock_name] = df

                    elif sheet_name.endswith('_估值'):
                        stock_name = sheet_name.replace('_估值', '')
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                        data['valuation_data'][stock_name] = df

                    elif '_' in sheet_name and not sheet_name.endswith(('_价格', '_估值')):
                        # 财务数据表
                        parts = sheet_name.split('_')
                        if len(parts) >= 2:
                            stock_name = parts[0]
                            table_name = '_'.join(parts[1:])
                            df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                            if stock_name not in data['financial_data']:
                                data['financial_data'][stock_name] = {}
                            data['financial_data'][stock_name][table_name] = df

                print(f"数据已从Excel文件 {self.data_file} 加载")
                return data
            else:
                print("数据文件不存在，将创建新数据")
                return None
        except Exception as e:
            print(f"从Excel加载数据失败: {e}")
            return None

    def update_data(self, force_update=False, start_date=None, end_date=None):
        """
        更新数据 - 这是主要的调用接口

        Parameters:
        force_update: 是否强制更新，即使数据是最新的也重新获取
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'
        """
        # 尝试加载现有数据
        existing_data = self.load_data()

        if existing_data and not force_update:
            if self.check_data_freshness(existing_data):
                print("数据已是最新，无需更新")
                return existing_data
            else:
                print("数据需要更新，开始收集新数据...")
        else:
            print("开始全量数据收集...")

        # 收集新数据
        new_data = self.collect_all_data(start_date, end_date)

        # 保存新数据
        if self.save_data(new_data):
            return new_data
        else:
            print("数据保存失败")
            return None

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

            if data_dict:
                valid_count = sum(1 for v in data_dict.values() if v is not None)
                total_count = len(data_dict)
                print(f"{data_type}: {valid_count}/{total_count} 只股票数据有效")
            else:
                print(f"{data_type}: 无数据")

def main():
    """
    主函数 - 支持自定义时间范围的数据收集
    用法:
    python stock_data_collector.py                          # 默认获取最近100个交易日
    python stock_data_collector.py --start 2024-01-01      # 从2024-01-01到今天
    python stock_data_collector.py --end 2024-12-31        # 从最早到2024-12-31
    python stock_data_collector.py --start 2024-01-01 --end 2024-12-31  # 指定时间范围
    """
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='股票数据收集器')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true', help='强制更新数据')

    args = parser.parse_args()

    # 创建数据收集器
    collector = StockDataCollector()

    # 认证
    if not collector.auth_jqdata():
        return

    # 更新数据（支持自定义时间范围）
    data = collector.update_data(
        force_update=args.force,
        start_date=args.start,
        end_date=args.end
    )

    # 显示数据摘要
    collector.get_data_summary(data)

    # 显示剩余调用次数
    print(f"\n剩余调用次数: {get_query_count()}")

if __name__ == "__main__":
    main()