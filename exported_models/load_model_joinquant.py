"""
聚宽平台 - 加载QuantaAlpha模型 (v4.0, 5因子)

前提条件：
1. 将 quantaalpha_model.txt 上传到聚宽平台（企业版）
2. 聚宽企业版支持文件上传和lightgbm库

使用步骤：
1. 上传 quantaalpha_model.txt 文件
2. 复制本脚本到策略代码
3. 运行回测

注意：聚宽平台不支持直接 open() 读取文件，
      必须使用 read_file() + tempfile + lgb.Booster 方式加载 .txt 模型。
"""

import lightgbm as lgb
import tempfile
import os
import numpy as np

BENCHMARK = "000300.XSHG"
INIT_CAPITAL = 1000000
TOP_K = 50
N_FACTORS = 5
MODEL_FILE = "quantaalpha_model.txt"

def load_model_official():
    """使用聚宽官方 API 加载 LightGBM .txt 模型"""
    import contextlib
    model_bytes = read_file(MODEL_FILE)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb')
    tmp_path = tmp.name
    try:
        with tmp:
            tmp.write(model_bytes)
            tmp.flush()
        model = lgb.Booster(model_file=tmp_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
    return model

def initialize(context):
    """初始化"""
    g.stock_pool = get_index_stocks(BENCHMARK)
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        min_commission=5
    ), type='stock')

    set_slippage(FixedSlippage(0.002))

    log.info("加载模型...")
    try:
        g.model = load_model_official()
        log.info("✅ 模型加载成功")
    except Exception as e:
        log.error(f"❌ 模型加载失败: {e}")
        g.model = None

    run_monthly(rebalance, 1)

def calculate_factors(stock, date):
    """计算5因子"""
    # ⚠️ count=100 确保60-day rank窗口有足够历史数据
    # (v4.0因子4: Turnover_Acceleration 需要 Rank(..., 60) + Delta(volume, 1) 的20日统计)
    df = get_price(stock, end_date=date, count=100, frequency='1d',
                   fields=['open', 'high', 'low', 'close', 'volume'])
    if df is None or len(df) < 80:
        return None

    close = df['close'].values
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # v4.0 五因子（因子库选优，覆盖5个驱动逻辑）
    try:
        returns = np.diff(close) / close[:-1]

        # 因子1: Gap_Range_Mean_Reversion_5D（均值回归类，ICIR=0.1610）
        if len(close) >= 10:
            gap = (open_price[-1] - close[-2]) / (np.mean(high[-10:] - low[-10:]) + 1e-8)
            range_ratio = (high[-1] - low[-1]) / (np.mean(high[-10:] - low[-10:]) + 1e-8)
            raw = gap * range_ratio
            raw_hist = [(open_price[i] - close[i-1]) / (np.mean(high[i-10:i] - low[i-10:i]) + 1e-8) * (high[i] - low[i]) / (np.mean(high[i-10:i] - low[i-10:i]) + 1e-8) for i in range(max(1, len(close)-10), len(close))]
            if len(raw_hist) >= 5:
                factor1 = (raw - np.mean(raw_hist[-5:])) / (np.std(raw_hist[-5:]) + 1e-8)
            else:
                factor1 = 0
        else:
            factor1 = 0

        # 因子2: Close_Position_Deviation_Factor_10D（动量/趋势类，ICIR=0.1610）
        if len(close) >= 10:
            close_pos = (close[-1] - low[-1]) / (high[-1] - low[-1] + 1e-8)
            deviation = abs(close_pos - 0.5)
            dev_hist = [abs((close[i] - low[i]) / (high[i] - low[i] + 1e-8) - 0.5) for i in range(max(1, len(close)-10), len(close))]
            factor2 = sum(1 for d in dev_hist if d < deviation) / len(dev_hist) if dev_hist else 0.5
        else:
            factor2 = 0

        # 因子3: Overnight_Intraday_Asymmetry_20D（其他类，ICIR=0.1610）
        if len(close) >= 20:
            overnight_ret = [(open_price[i] - close[i-1]) / (close[i-1] + 1e-8) for i in range(1, len(close))]
            intraday_pos = [(close[i] - low[i]) / (high[i] - low[i] + 1e-8) for i in range(len(close))]
            n = min(20, len(overnight_ret), len(intraday_pos))
            if n >= 5:
                o = overnight_ret[-n:]
                i = intraday_pos[-n:]
                mean_o = np.mean(o)
                mean_i = np.mean(i)
                cov = np.mean([(o[j]-mean_o)*(i[j]-mean_i) for j in range(n)])
                std_o = np.std(o)
                std_i = np.std(i)
                factor3 = cov / (std_o * std_i + 1e-8)
            else:
                factor3 = 0
        else:
            factor3 = 0

        # 因子4: Turnover_Acceleration_10D_vs_20D（量价关系类，ICIR=0.1569）
        # Qlib表达式: Rank((Mean(Delta($volume, 1), 10) - Mean(Delta($volume, 1), 20)) / (Std(Delta($volume, 1), 20) + 1e-8), 60)
        # 含义: 每天的accel = (vol_delta的10日均值 - 20日均值) / 20日std
        #       然后Rank(accel_today, 过去60天的accel历史)
        if len(volume) >= 80:
            vol_delta = np.diff(volume)
            # 计算每天的acceleration值（需要20日窗口）
            accel_series = []
            for t in range(20, len(vol_delta) + 1):
                window = vol_delta[t-20:t]
                m10 = np.mean(window[-10:])
                m20 = np.mean(window)
                s20 = np.std(window)
                accel_series.append((m10 - m20) / (s20 + 1e-8))
            # 取最后60个accel值作为rank历史
            accel_history = accel_series[-60:] if len(accel_series) >= 60 else accel_series
            # 当前accel（最后一天）
            accel_cur = accel_series[-1]
            # Rank: 当前值在历史中的百分位
            factor4 = sum(1 for a in accel_history if a < accel_cur) / len(accel_history) if accel_history else 0.5
        else:
            factor4 = 0

        # 因子5: Abnormal_Return_Persistence_Score（波动率类，ICIR=0.1370）
        if len(returns) >= 20:
            mean_ret = np.mean(returns[-20:])
            std_ret = np.std(returns[-20:])
            threshold = 1.5 * std_ret
            abnormal_count = sum(1 for r in returns[-5:] if abs(r - mean_ret) > threshold)
            factor5 = abnormal_count / 5.0
        else:
            factor5 = 0

        return np.array([factor1, factor2, factor3, factor4, factor5])
    except:
        return None

def rebalance(context):
    """每月调仓"""
    if g.model is None:
        return

    log.info(f"调仓: {context.current_dt.date()}")

    predictions = {}
    for stock in g.stock_pool:
        factors = calculate_factors(stock, context.current_dt)
        if factors is not None:
            pred = g.model.predict(factors.reshape(1, -1))[0]
            predictions[stock] = pred

    if len(predictions) < TOP_K:
        return

    top_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_stocks = [s[0] for s in top_stocks]

    for stock in context.portfolio.positions:
        if stock not in top_stocks:
            order_target(stock, 0)

    weight = 1.0 / TOP_K
    for stock in top_stocks:
        order_target_value(stock, context.portfolio.total_value * weight)

    log.info(f"调仓完成")

def handle_data(context, data):
    pass

def after_code_changed(context):
    """代码修改后重新加载模型（聚宽热更新回调）"""
    log.info("代码已更新，重新加载模型...")
    try:
        g.model = load_model_official()
        log.info("✅ 模型重新加载成功")
    except Exception as e:
        log.error(f"❌ 模型重新加载失败: {e}")
        g.model = None
