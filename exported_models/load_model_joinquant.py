"""
聚宽平台 - 加载QuantaAlpha模型

前提条件：
1. 将 quantaalpha_model.pkl 上传到聚宽平台
2. 聚宽企业版支持文件上传和lightgbm库

使用步骤：
1. 上传模型文件
2. 复制本脚本到策略代码
3. 运行回测
"""

import pickle
import numpy as np

BENCHMARK = "000300.XSHG"
INIT_CAPITAL = 1000000
TOP_K = 50

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
        # 加载上传的模型文件
        with open('quantaalpha_model.pkl', 'rb') as f:
            g.model = pickle.load(f)
        log.info("✅ 模型加载成功")
    except Exception as e:
        log.error(f"❌ 模型加载失败: {e}")
        g.model = None

    run_monthly(rebalance, 1)

def calculate_factors(stock, date):
    """计算三因子"""
    df = get_price(stock, end_date=date, count=50, frequency='1d', fields=['close'])
    if df is None or len(df) < 30:
        return None

    close = df['close'].values

    try:
        # 因子1: Hurst Proxy
        log_close = np.log(close)
        zscore1 = (log_close[-1] - np.mean(log_close[-20:])) / (np.std(log_close[-20:]) + 1e-8)
        returns = np.diff(close) / close[:-1]
        vol_short = np.std(returns[-5:])
        vol_long = np.std(returns[-20:])
        factor1 = -zscore1 * (vol_short / (vol_long + 1e-8))

        # 因子2: AR1 Reversion
        zscore2 = (close[-1] - np.mean(close[-20:])) / (np.std(close[-20:]) + 1e-8)
        if len(returns) >= 20:
            corr = np.corrcoef(returns[-20:-1], returns[-19:])[0, 1]
        else:
            corr = 0
        factor2 = -zscore2 * np.sign(-corr)

        # 因子3: OU MeanReversion
        if len(returns) >= 10:
            corr_short = np.corrcoef(returns[-10:-1], returns[-9:])[0, 1]
        else:
            corr_short = 0
        factor3 = -zscore2 * (1 - corr_short)

        return np.array([factor1, factor2, factor3])
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
