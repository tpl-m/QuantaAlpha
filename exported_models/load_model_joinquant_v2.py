"""
聚宽平台 - QuantaAlpha v2.0 加载脚本 (9因子, Optuna联合搜索版本)

来源: Optuna Trial #49, Score=-0.0305
因子数: 9 (覆盖量价关系、动量/趋势、波动率)

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
import contextlib
import numpy as np

BENCHMARK = "000300.XSHG"
INIT_CAPITAL = 1000000
TOP_K = 50
N_FACTORS = 9
MODEL_FILE = "quantaalpha_model.txt"

# 聚宽特殊限制: count必须>=80（因子2/5/6/7/8/9需要60日rolling window + 20日统计）
MIN_HISTORY = 100


def load_model_official():
    """使用聚宽官方 API 加载 LightGBM .txt 模型

    ⚠️ 坑点13教训：聚宽不能用open()读取文件，必须用read_file() + tempfile
    """
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

    log.info("加载QuantaAlpha v2.0模型...")
    try:
        g.model = load_model_official()
        log.info(f"✅ 模型加载成功 ({g.model.num_trees()} trees)")
    except Exception as e:
        log.error(f"❌ 模型加载失败: {e}")
        g.model = None

    run_monthly(rebalance, 1)


def calculate_factors(stock, date):
    """计算9个因子值

    ⚠️ 坑点13教训：get_price的fields必须包含open/high/low/close/volume全部字段
    ⚠️ count=100 确保60-day rank窗口有足够历史数据
    """
    # ⚠️ 必须获取全部OHLCV字段
    df = get_price(stock, end_date=date, count=MIN_HISTORY, frequency='1d',
                   fields=['open', 'high', 'low', 'close', 'volume'])
    if df is None or len(df) < 80:
        return None

    close = df['close'].values
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    try:
        returns = np.diff(close) / close[:-1]

        # ============================================================
        # 因子1: Close_Position_Deviation_Factor_10D
        # Qlib: Rank(Abs(($close - $low) / ($high - $low + 1e-8) - 0.5), 10)
        # ============================================================
        if len(close) >= 10:
            close_pos = (close[-1] - low[-1]) / (high[-1] - low[-1] + 1e-8)
            deviation = abs(close_pos - 0.5)
            # Rank in last 10 days
            dev_hist = [abs((close[i] - low[i]) / (high[i] - low[i] + 1e-8) - 0.5)
                       for i in range(max(0, len(close)-10), len(close))]
            factor1 = sum(1 for d in dev_hist if d < deviation) / len(dev_hist) if dev_hist else 0.5
        else:
            factor1 = 0

        # ============================================================
        # 因子2: Turnover_Acceleration_10D_vs_20D
        # Qlib: Rank((Mean(Delta($volume,1),10)-Mean(Delta($volume,1),20))/(Std(Delta($volume,1),20)+1e-8), 60)
        # ============================================================
        if len(volume) >= 80:
            vol_delta = np.diff(volume)
            accel_series = []
            for t in range(20, len(vol_delta) + 1):
                w = vol_delta[t-20:t]
                m10 = np.mean(w[-10:])
                m20 = np.mean(w)
                s20 = np.std(w)
                accel_series.append((m10 - m20) / (s20 + 1e-8))
            accel_history = accel_series[-60:] if len(accel_series) >= 60 else accel_series
            accel_cur = accel_series[-1]
            factor2 = sum(1 for a in accel_history if a < accel_cur) / len(accel_history) if accel_history else 0.5
        else:
            factor2 = 0

        # ============================================================
        # 因子3: Price_Volume_Rank_Divergence_10D
        # Qlib: Rank(Delta($close,10),60) - Rank(Delta($volume,10),60)
        # ============================================================
        if len(close) >= 70:  # need 60+10 for rolling
            close_delta = close[-1] - close[-11] if len(close) >= 11 else 0
            vol_delta_val = volume[-1] - volume[-11] if len(volume) >= 11 else 0
            # Rank over last 60 days
            close_deltas = [close[i] - close[i-10] for i in range(10, len(close))]
            vol_deltas = [volume[i] - volume[i-10] for i in range(10, len(volume))]
            close_deltas_60 = close_deltas[-60:] if len(close_deltas) >= 60 else close_deltas
            vol_deltas_60 = vol_deltas[-60:] if len(vol_deltas) >= 60 else vol_deltas
            rank_close = sum(1 for x in close_deltas_60 if x < close_delta) / len(close_deltas_60) if close_deltas_60 else 0.5
            rank_vol = sum(1 for x in vol_deltas_60 if x < vol_delta_val) / len(vol_deltas_60) if vol_deltas_60 else 0.5
            factor3 = rank_close - rank_vol
        else:
            factor3 = 0

        # ============================================================
        # 因子4: Liquidity_Adjusted_Divergence_10D
        # Qlib: Rank(Delta($close,10),60) - Rank(Delta($volume,10),60) * Rank(Mean($close*$volume,20),60)
        # ============================================================
        if len(close) >= 80:
            close_deltas = [close[i] - close[i-10] for i in range(10, len(close))]
            vol_deltas = [volume[i] - volume[i-10] for i in range(10, len(volume))]
            cv_means = [np.mean(close[i-20:i] * volume[i-20:i]) for i in range(20, len(close))]

            close_deltas_60 = close_deltas[-60:] if len(close_deltas) >= 60 else close_deltas
            vol_deltas_60 = vol_deltas[-60:] if len(vol_deltas) >= 60 else vol_deltas
            cv_means_60 = cv_means[-60:] if len(cv_means) >= 60 else cv_means

            close_delta_cur = close[-1] - close[-11]
            vol_delta_cur = volume[-1] - volume[-11]
            cv_mean_cur = np.mean(close[-20:] * volume[-20:])

            r_close = sum(1 for x in close_deltas_60 if x < close_delta_cur) / len(close_deltas_60) if close_deltas_60 else 0.5
            r_vol = sum(1 for x in vol_deltas_60 if x < vol_delta_cur) / len(vol_deltas_60) if vol_deltas_60 else 0.5
            r_cv = sum(1 for x in cv_means_60 if x < cv_mean_cur) / len(cv_means_60) if cv_means_60 else 0.5

            factor4 = r_close - r_vol * r_cv
        else:
            factor4 = 0

        # ============================================================
        # 因子5: Intraday_Overnight_Volatility_Ratio_10D
        # Qlib: Rank(Mean(($high-$low)/($open+1e-8),10)/(Mean(Abs($open/Ref($close,1)-1),10)+1e-8), 60)
        # ============================================================
        if len(close) >= 70:
            ratios = []
            for i in range(10, len(close)):
                intraday_vol = np.mean([(high[j] - low[j]) / (open_price[j] + 1e-8)
                                       for j in range(i-10, i)])
                overnight_vol = np.mean([abs(open_price[j] / (close[j-1] + 1e-8) - 1)
                                        for j in range(i-10, i)])
                ratios.append(intraday_vol / (overnight_vol + 1e-8))
            ratios_60 = ratios[-60:] if len(ratios) >= 60 else ratios
            cur_ratio = ratios[-1] if ratios else 0
            factor5 = sum(1 for r in ratios_60 if r < cur_ratio) / len(ratios_60) if ratios_60 else 0.5
        else:
            factor5 = 0

        # ============================================================
        # 因子6: Volume_Weighted_Intraday_Movement_15D
        # Qlib: Rank(Mean((Abs($close-$open)*$volume)/(Abs($open-Ref($close,1))+1e-8),15), 60)
        # ============================================================
        if len(close) >= 75:
            movements = []
            for i in range(15, len(close)):
                mv = np.mean([(abs(close[j] - open_price[j]) * volume[j]) /
                             (abs(open_price[j] - close[j-1]) + 1e-8)
                             for j in range(i-15, i)])
                movements.append(mv)
            mov_60 = movements[-60:] if len(movements) >= 60 else movements
            cur_mov = movements[-1] if movements else 0
            factor6 = sum(1 for m in mov_60 if m < cur_mov) / len(mov_60) if mov_60 else 0.5
        else:
            factor6 = 0

        # ============================================================
        # 因子7: Asymmetric_Volume_Concentration_20D
        # Qlib: Rank(Mean(($close-$open)*$vol/($high-$low+1e-8),20)-Mean(($open-Ref($close,1))*$vol/($high-$low+1e-8),20), 60)
        # ============================================================
        if len(close) >= 80:
            asymmetries = []
            for i in range(20, len(close)):
                intraday = np.mean([(close[j] - open_price[j]) * volume[j] / (high[j] - low[j] + 1e-8)
                                   for j in range(i-20, i)])
                overnight = np.mean([(open_price[j] - close[j-1]) * volume[j] / (high[j] - low[j] + 1e-8)
                                    for j in range(i-20, i)])
                asymmetries.append(intraday - overnight)
            asym_60 = asymmetries[-60:] if len(asymmetries) >= 60 else asymmetries
            cur_asym = asymmetries[-1] if asymmetries else 0
            factor7 = sum(1 for a in asym_60 if a < cur_asym) / len(asym_60) if asym_60 else 0.5
        else:
            factor7 = 0

        # ============================================================
        # 因子8: Normalized_Rank_Stability_Volume_Confirmation_Factor
        # Qlib: Rank(Corr($return,Ref($return,15),20),60) * Rank(Mean($vol,5)/Mean($vol,20),60)
        # ============================================================
        if len(close) >= 80:
            # Corr part
            corrs = []
            for i in range(20, len(close)):
                r1 = [(close[j] - close[j-1]) / (close[j-1] + 1e-8) for j in range(i-20, i)]
                r2 = [(close[j] - close[j-15]) / (close[j-15] + 1e-8) for j in range(i-20, i)]
                if np.std(r1) > 0 and np.std(r2) > 0:
                    corr = np.corrcoef(r1, r2)[0, 1]
                    if not np.isnan(corr):
                        corrs.append(corr)
                    else:
                        corrs.append(0)
                else:
                    corrs.append(0)

            # Volume ratio part
            vol_ratios = []
            for i in range(20, len(close)):
                v5 = np.mean(volume[i-5:i])
                v20 = np.mean(volume[i-20:i])
                vol_ratios.append(v5 / (v20 + 1e-8))

            corrs_60 = corrs[-60:] if len(corrs) >= 60 else corrs
            vr_60 = vol_ratios[-60:] if len(vol_ratios) >= 60 else vol_ratios

            cur_corr = corrs[-1] if corrs else 0
            cur_vr = vol_ratios[-1] if vol_ratios else 0.5

            r_corr = sum(1 for c in corrs_60 if c < cur_corr) / len(corrs_60) if corrs_60 else 0.5
            r_vr = sum(1 for v in vr_60 if v < cur_vr) / len(vr_60) if vr_60 else 0.5

            factor8 = r_corr * r_vr
        else:
            factor8 = 0

        # ============================================================
        # 因子9: Cross_Sectional_Momentum_Acceleration_Volume_Factor
        # Qlib: ((Rank(Sum($return,5),60)-Rank(Sum($return,20),60))*(Mean($vol,5)/Mean($vol,20))
        #       - Mean(...,20)) / (Std(...,20)+1e-8)  -- TS_ZSCORE of momentum*volume
        # ============================================================
        if len(close) >= 80:
            momentum_vol_products = []
            for i in range(20, len(close)):
                sum5 = sum((close[j] - close[j-1]) / (close[j-1] + 1e-8) for j in range(i-5, i))
                sum20 = sum((close[j] - close[j-1]) / (close[j-1] + 1e-8) for j in range(i-20, i))
                v5 = np.mean(volume[i-5:i])
                v20 = np.mean(volume[i-20:i])
                momentum_vol_products.append((sum5 - sum20) * (v5 / (v20 + 1e-8)))

            mvp_20 = momentum_vol_products[-20:] if len(momentum_vol_products) >= 20 else momentum_vol_products
            cur_mvp = momentum_vol_products[-1] if momentum_vol_products else 0

            mean_20 = np.mean(mvp_20)
            std_20 = np.std(mvp_20)
            factor9 = (cur_mvp - mean_20) / (std_20 + 1e-8)
        else:
            factor9 = 0

        return np.array([factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8, factor9])
    except Exception as e:
        log.warning(f"因子计算失败 {stock}: {e}")
        return None


def rebalance(context):
    """每月调仓"""
    if g.model is None:
        log.warning("模型未加载，跳过调仓")
        return

    log.info(f"调仓: {context.current_dt.date()}")

    predictions = {}
    for stock in g.stock_pool:
        factors = calculate_factors(stock, context.current_dt)
        if factors is not None:
            pred = g.model.predict(factors.reshape(1, -1))[0]
            predictions[stock] = pred

    if len(predictions) < TOP_K:
        log.warning(f"预测股票不足 {TOP_K} 只，跳过调仓")
        return

    top_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_stocks = [s[0] for s in top_stocks]

    # 卖出不在目标列表的股票
    for stock in list(context.portfolio.positions.keys()):
        if stock not in top_stocks:
            order_target(stock, 0)

    # 等权买入目标股票
    weight = 1.0 / TOP_K
    for stock in top_stocks:
        order_target_value(stock, context.portfolio.total_value * weight)

    log.info(f"调仓完成: {len(top_stocks)} 只股票")


def handle_data(context, data):
    pass


def after_code_changed(context):
    """代码修改后重新加载模型（聚宽热更新回调）"""
    log.info("代码已更新，重新加载模型...")
    try:
        g.model = load_model_official()
        log.info(f"✅ 模型重新加载成功 ({g.model.num_trees()} trees)")
    except Exception as e:
        log.error(f"❌ 模型重新加载失败: {e}")
        g.model = None
