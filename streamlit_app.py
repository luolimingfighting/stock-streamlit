import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# XGBoost暂时注释掉，因为需要OpenMP运行时
# from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# LSTM相关导入
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("LSTM功能不可用，需要安装tensorflow")

# 页面配置
st.set_page_config(
    page_title="股票预测系统",
    page_icon="📈",
    layout="wide"
)

# 初始化session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def fetch_stock_data(stock_code, start_date, end_date):
    """获取股票历史数据"""
    try:
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                      start_date=start_date, end_date=end_date,
                                      adjust="qfq")
        if stock_data.empty:
            return None

        # 检查列数并动态调整
        expected_columns = ['日期', '开盘价', '收盘价', '最高价', '最低价',
                          '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

        # akashare返回12列，我们只需要前11列（去掉股票代码列）
        if len(stock_data.columns) == 12:
            # 删除第二列（股票代码）
            stock_data = stock_data.drop(stock_data.columns[1], axis=1)
            stock_data.columns = expected_columns
        elif len(stock_data.columns) == 11:
            stock_data.columns = expected_columns
        else:
            st.warning(f"未知的数据格式，列数: {len(stock_data.columns)}")
            # 尝试使用前11列
            stock_data = stock_data.iloc[:, :11]
            stock_data.columns = expected_columns[:len(stock_data.columns)]

        # 转换数据类型
        numeric_columns = ['开盘价', '收盘价', '最高价', '最低价', '成交量',
                          '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        for col in numeric_columns:
            if col in stock_data.columns:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

        stock_data['日期'] = pd.to_datetime(stock_data['日期'])
        stock_data = stock_data.sort_values('日期').reset_index(drop=True)

        return stock_data
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None

def create_features(data):
    """创建技术指标特征"""
    df = data.copy()

    # 移动平均线
    df['MA5'] = df['收盘价'].rolling(window=5).mean()
    df['MA10'] = df['收盘价'].rolling(window=10).mean()
    df['MA20'] = df['收盘价'].rolling(window=20).mean()

    # 相对强弱指数 (RSI)
    delta = df['收盘价'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 布林带
    df['BB_middle'] = df['收盘价'].rolling(window=20).mean()
    bb_std = df['收盘价'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std

    # MACD
    exp12 = df['收盘价'].ewm(span=12, adjust=False).mean()
    exp26 = df['收盘价'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # 价格变化率
    df['Price_Change'] = df['收盘价'].pct_change()

    # 成交量变化率
    df['Volume_Change'] = df['成交量'].pct_change()

    return df.dropna()
def prepare_lstm_data(data, sequence_length=30):
    """为LSTM准备时间序列数据"""
    # 使用收盘价作为主要特征
    prices = data['收盘价'].values.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    # 创建时间序列数据集
    X, y = [], []
    for i in range(len(scaled_prices) - sequence_length):
        X.append(scaled_prices[i:i+sequence_length])
        y.append(scaled_prices[i+sequence_length])

    return np.array(X), np.array(y), scaler

def train_lstm_model(data, sequence_length=30, epochs=50, batch_size=32):
    """训练LSTM模型"""
    X, y, scaler = prepare_lstm_data(data, sequence_length)

    # 分割数据
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 构建LSTM模型
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='mean_squared_error')

    # 训练模型
    history = model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       verbose=0)

    # 预测
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    # 评估指标
    mae = mean_absolute_error(y_test_actual, y_pred)
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred)

    return model, X_test, y_test_actual, y_pred, scaler, {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def train_model(data, model_type='random_forest', test_size=0.2):
    """训练预测模型"""
    # 如果是LSTM模型，使用专门的训练函数
    if model_type == 'lstm' and LSTM_AVAILABLE:
        return train_lstm_model(data)

    # 准备特征和目标
    features = ['开盘价', '最高价', '最低价', '成交量', 'MA5', 'MA10', 'MA20',
               'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal',
               'MACD_hist', 'Price_Change', 'Volume_Change']
    target = '收盘价'

    X = data[features]
    y = data[target]

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # 选择模型
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'xgboost':
        st.warning("XGBoost功能暂时不可用，需要安装OpenMP运行时。使用随机森林代替。")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def predict_future(model, last_data, days=30):
    """预测未来价格"""
    predictions = []
    current_features = last_data.copy()

    for _ in range(days):
        # 预测下一天
        pred = model.predict([current_features])[0]
        predictions.append(pred)

        # 更新特征（这里简化处理，实际需要更复杂的特征更新逻辑）
        current_features = update_features(current_features, pred)

    return predictions

def update_features(features, new_price):
    """更新特征用于连续预测（简化版本）"""
    # 这里需要实现更复杂的特征更新逻辑
    return features

def predict_lstm_future(model, data, scaler, days=30, sequence_length=30):
    """使用LSTM预测未来价格"""
    # 获取最近sequence_length天的数据
    recent_prices = data['收盘价'].values[-sequence_length:]
    recent_scaled = scaler.transform(recent_prices.reshape(-1, 1))

    predictions = []
    current_sequence = recent_scaled.copy()

    for _ in range(days):
        # 预测下一天
        pred_scaled = model.predict(current_sequence.reshape(1, sequence_length, 1), verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)

        # 更新序列
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred_scaled[0][0]

    return predictions

def main():
    st.title("📈 股票预测系统")

    # 侧边栏
    st.sidebar.header("股票选择和设置")

    # 股票搜索和选择
    stock_code = st.sidebar.text_input("股票代码（例如：000001）", "000001")
    stock_name = st.sidebar.text_input("股票名称（可选）", "")

    # 时间范围选择
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("开始日期", pd.to_datetime("2023-01-01"))
    with col2:
        end_date = st.date_input("结束日期", pd.to_datetime("2024-01-01"))

        # 模型选择
    model_options = ["random_forest", "linear_regression"]
    if LSTM_AVAILABLE:
        model_options.append("lstm")
    model_options.append("xgboost")

    model_type = st.sidebar.selectbox(
        "选择预测模型",
        model_options,
        index=0
    )

    # 预测天数
    predict_days = st.sidebar.slider("预测天数", 5, 60, 30)

    # 获取数据按钮
    if st.sidebar.button("获取数据并分析"):
        with st.spinner("正在获取股票数据..."):
            stock_data = fetch_stock_data(stock_code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

            if stock_data is not None:
                st.session_state.stock_data = stock_data
                st.success("数据获取成功！")
            else:
                st.error("获取数据失败，请检查股票代码是否正确")

    # 主内容区
    if st.session_state.stock_data is not None:
        stock_data = st.session_state.stock_data

        # 显示股票基本信息
        st.header("股票基本信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最新收盘价", f"{stock_data['收盘价'].iloc[-1]:.2f}")
        with col2:
            change = stock_data['收盘价'].iloc[-1] - stock_data['收盘价'].iloc[-2]
            st.metric("涨跌额", f"{change:.2f}")
        with col3:
            change_pct = (change / stock_data['收盘价'].iloc[-2]) * 100
            st.metric("涨跌幅", f"{change_pct:.2f}%")
        with col4:
            st.metric("成交量", f"{stock_data['成交量'].iloc[-1]:,.0f}")

        # 价格图表
        st.header("价格走势图")
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('价格走势', '成交量'),
                           vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])

        # 价格图表
        fig.add_trace(go.Candlestick(
            x=stock_data['日期'],
            open=stock_data['开盘价'],
            high=stock_data['最高价'],
            low=stock_data['最低价'],
            close=stock_data['收盘价'],
            name='OHLC'
        ), row=1, col=1)

        # 成交量图表
        fig.add_trace(go.Bar(
            x=stock_data['日期'],
            y=stock_data['成交量'],
            name='成交量',
            marker_color='rgba(0,0,255,0.3)'
        ), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 特征工程和模型训练
        st.header("模型训练和预测")

        with st.spinner("正在创建特征和训练模型..."):
            feature_data = create_features(stock_data)

            if len(feature_data) > 50:  # 确保有足够的数据
                if model_type == 'lstm' and LSTM_AVAILABLE:
                    model, X_test, y_test, y_pred, scaler, metrics = train_model(feature_data, model_type)
                    st.session_state.lstm_scaler = scaler
                else:
                    model, X_test, y_test, y_pred, metrics = train_model(feature_data, model_type)
                st.session_state.model = model
                st.session_state.model = model

                # 显示评估指标
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{metrics['MAE']:.4f}")
                col2.metric("MSE", f"{metrics['MSE']:.4f}")
                col3.metric("RMSE", f"{metrics['RMSE']:.4f}")
                col4.metric("R²", f"{metrics['R2']:.4f}")

                # 预测结果图表
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=feature_data['日期'][-len(y_test):],
                    y=y_test.values,
                    name='实际价格',
                    line=dict(color='blue')
                ))
                fig_pred.add_trace(go.Scatter(
                    x=feature_data['日期'][-len(y_test):],
                    y=y_pred,
                    name='预测价格',
                    line=dict(color='red')
                ))
                fig_pred.update_layout(
                    title='模型预测效果',
                    xaxis_title='日期',
                    yaxis_title='价格'
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # 未来预测
                if st.button("预测未来价格"):
                    with st.spinner("正在进行未来预测..."):
                        if model_type == 'lstm' and LSTM_AVAILABLE:
                            # LSTM专用预测
                            future_predictions = predict_lstm_future(model, feature_data, st.session_state.lstm_scaler, predict_days)
                        else:
                            last_features = feature_data.iloc[-1][[
                                '开盘价', '最高价', '最低价', '成交量', 'MA5', 'MA10', 'MA20',
                                'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal',
                                'MACD_hist', 'Price_Change', 'Volume_Change'
                            ]].values
                            future_predictions = predict_future(model, last_features, predict_days)
                        future_dates = pd.date_range(
                            start=feature_data['日期'].iloc[-1] + pd.Timedelta(days=1),
                            periods=predict_days
                        )

                        # 显示预测结果
                        st.subheader("未来价格预测")
                        pred_df = pd.DataFrame({
                            '日期': future_dates,
                            '预测价格': future_predictions
                        })
                        st.dataframe(pred_df)

                        # 预测图表
                        fig_future = go.Figure()
                        fig_future.add_trace(go.Scatter(
                            x=feature_data['日期'][-30:],  # 显示最近30天
                            y=feature_data['收盘价'][-30:],
                            name='历史价格',
                            line=dict(color='blue')
                        ))
                        fig_future.add_trace(go.Scatter(
                            x=pred_df['日期'],
                            y=pred_df['预测价格'],
                            name='预测价格',
                            line=dict(color='green')
                        ))
                        fig_future.update_layout(
                            title='未来价格预测',
                            xaxis_title='日期',
                            yaxis_title='价格'
                        )
                        st.plotly_chart(fig_future, use_container_width=True)
            else:
                st.warning("数据量不足，无法进行有效的模型训练")

    else:
        st.info("请在侧边栏输入股票代码并点击'获取数据并分析'开始使用")

if __name__ == "__main__":
    main()