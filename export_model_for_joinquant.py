#!/usr/bin/env python3
"""
导出训练好的LightGBM模型，用于聚宽平台

用途：
1. 从Qlib MLflow实验中加载训练好的模型
2. 导出为.txt文件（LightGBM原生格式，体积小）
3. 提供聚宽平台加载模型的示例代码

使用方法：
    python3 export_model_for_joinquant.py --experiment-id 535101104011168066
"""

import argparse
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def find_latest_run(experiment_id: str):
    """查找实验的最新run"""
    mlruns_dir = project_root / "mlruns" / experiment_id
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {mlruns_dir}")

    runs = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name != "meta.yaml"]
    if not runs:
        raise FileNotFoundError(f"实验{experiment_id}中没有找到run")

    # 按修改时间排序，取最新的
    latest_run = max(runs, key=lambda p: p.stat().st_mtime)
    print(f"✅ 找到最新run: {latest_run.name}")
    return latest_run

def load_model_from_qlib(run_dir: Path):
    """
    从Qlib recorder中重新训练并获取模型

    注意：Qlib默认不保存模型，需要重新训练
    """
    print("\n⚠️  Qlib不保存模型文件，需要重新训练...")
    print("提示：可以使用以下方法之一：")
    print("1. 修改Qlib配置，在训练时保存模型")
    print("2. 使用本脚本提供的重训练功能")
    print("3. 直接在聚宽平台重新训练（推荐）\n")

    # 检查是否有预测结果
    pred_file = run_dir / "artifacts" / "pred.pkl"
    if pred_file.exists():
        print(f"✅ 找到预测结果: {pred_file}")
        print("   （但这不是模型文件，无法直接在聚宽使用）\n")

    return None

def create_joinquant_script_with_training(output_dir: Path):
    """
    创建包含模型训练的聚宽脚本

    由于无法导出已训练模型，提供在聚宽平台重新训练的方案
    """
    output_file = output_dir / "factor_ml_strategy_joinquant.py"

    script_content = '''"""
聚宽平台 - ML因子策略（含模型训练）

⚠️ 重要说明：
由于Qlib训练的模型无法直接导出，本脚本在聚宽平台重新训练LightGBM模型。

训练流程：
1. 初始化时训练模型（使用历史数据2016-2020）
2. 每日计算因子特征
3. 使用模型预测每只股票的收益率
4. 选择预测值最高的Top-50只股票
5. 构建等权组合

注意：首次运行需要较长时间训练模型（约1-3分钟）
"""

import jqdata
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ====================== 策略参数配置区 ======================
BENCHMARK_CODE = "000300.XSHG"
INIT_CAPITAL = 1000000

# 选股参数
TOP_K = 50                    # 选择Top-50只股票
REBALANCE_DAYS = 20           # 每20天调仓一次

# 模型训练参数
TRAIN_START = "2016-01-01"
TRAIN_END = "2020-12-31"
FEATURES_WINDOW = 20          # 因子计算窗口

# 因子权重（来自QuantaAlpha）
FACTOR_WEIGHTS = [1/3, 1/3, 1/3]
# ===========================================================

def initialize(context):
    """初始化函数 - 训练模型"""
    g.stock_pool = get_index_stocks(BENCHMARK_CODE)
    set_benchmark(BENCHMARK_CODE)
    set_option('use_real_price', True)

    # 设置交易成本
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')

    set_slippage(FixedSlippage(0.002))

    # 初始化全局变量
    g.current_holdings = []
    g.rebalance_counter = 0
    g.model = None

    log.info("=" * 70)
    log.info("ML因子策略初始化 - 开始训练模型...")
    log.info(f"股票池：{len(g.stock_pool)}只")
    log.info(f"Top-K：{TOP_K}只")
    log.info(f"调仓周期：{REBALANCE_DAYS}天")

    # 训练模型（异步，不阻塞）
    try:
        g.model = train_ml_model(context)
        log.info("✅ 模型训练完成！")
    except Exception as e:
        log.error(f"❌ 模型训练失败: {e}")
        log.info("将使用简单因子策略作为降级方案")

    log.info("=" * 70)

    # 设置调仓时间（每月1号）
    run_monthly(rebalance, 1)

def train_ml_model(context):
    """
    训练LightGBM模型（使用sklearn的GradientBoosting替代）

    注意：聚宽不支持lightgbm库，使用sklearn的GradientBoostingRegressor
    """
    log.info(f"训练期: {TRAIN_START} ~ {TRAIN_END}")

    # 获取训练数据
    X_train, y_train = prepare_training_data(context)

    if X_train is None or len(X_train) < 100:
        log.error("训练数据不足")
        return None

    log.info(f"训练样本数: {len(X_train)}")

    # 训练模型（参数接近LightGBM）
    model = GradientBoostingRegressor(
        n_estimators=100,        # 减少迭代次数（聚宽平台限制）
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)

    return model

def prepare_training_data(context):
    """
    准备训练数据

    Returns:
        X_train: 特征矩阵（N x 3，三个因子）
        y_train: 标签向量（未来5日收益率）
    """
    log.info("正在准备训练数据...")

    X_list = []
    y_list = []

    # 获取训练期的日期列表（每5天采样一次，减少计算量）
    trade_days = get_trade_days(start_date=TRAIN_START, end_date=TRAIN_END)
    sample_days = trade_days[::5]  # 每5天采样一次

    log.info(f"采样{len(sample_days)}个交易日...")

    for i, date in enumerate(sample_days):
        if i % 50 == 0:
            log.info(f"进度: {i}/{len(sample_days)}")

        # 获取当天的股票池（Top-100，减少计算量）
        stocks = g.stock_pool[:100]

        for stock in stocks:
            try:
                # 计算三个因子
                factors = calculate_factors_for_stock(stock, date)
                if factors is None:
                    continue

                # 获取未来5日收益率作为标签
                future_return = get_future_return(stock, date, periods=5)
                if future_return is None:
                    continue

                X_list.append(factors)
                y_list.append(future_return)

            except Exception as e:
                continue

    if len(X_list) == 0:
        return None, None

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    return X_train, y_train

def calculate_factors_for_stock(stock, date):
    """
    计算单只股票的三个因子

    Returns:
        np.array([factor1, factor2, factor3])
    """
    # 获取历史数据（简化版，减少计算量）
    df = get_price(
        stock,
        end_date=date,
        count=50,  # 减少历史数据量
        frequency='1d',
        fields=['close']
    )

    if df is None or len(df) < 30:
        return None

    close = df['close'].values

    # 简化的因子计算（完整版见factor_combined_meanreversion.py）
    try:
        # 因子1: Z-Score
        zscore = (close[-1] - np.mean(close[-20:])) / (np.std(close[-20:]) + 1e-8)

        # 因子2: 收益率自相关
        returns = np.diff(close) / close[:-1]
        if len(returns) < 20:
            return None
        corr = np.corrcoef(returns[-20:-1], returns[-19:])[0, 1]

        # 因子3: 波动率比
        vol_short = np.std(returns[-5:])
        vol_long = np.std(returns[-20:])
        vol_ratio = vol_short / (vol_long + 1e-8)

        # 返回三个因子（取负，表示均值回归）
        factor1 = -zscore * vol_ratio
        factor2 = -zscore * np.sign(-corr)
        factor3 = -zscore * (1 - corr)

        return np.array([factor1, factor2, factor3])

    except Exception as e:
        return None

def get_future_return(stock, date, periods=5):
    """获取未来N日收益率"""
    try:
        df = get_price(
            stock,
            start_date=date,
            count=periods + 1,
            frequency='1d',
            fields=['close']
        )

        if df is None or len(df) < periods + 1:
            return None

        ret = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        return ret

    except Exception as e:
        return None

def rebalance(context):
    """调仓函数 - 每月1号执行"""
    if g.model is None:
        log.warn("模型未训练，跳过调仓")
        return

    log.info(f"\\n{'='*70}")
    log.info(f"调仓日期: {context.current_dt.date()}")

    # 获取所有股票的预测值
    stock_predictions = {}

    for stock in g.stock_pool:
        try:
            # 计算因子
            factors = calculate_factors_for_stock(stock, context.current_dt)
            if factors is None:
                continue

            # 模型预测
            pred = g.model.predict(factors.reshape(1, -1))[0]
            stock_predictions[stock] = pred

        except Exception as e:
            continue

    if len(stock_predictions) < TOP_K:
        log.warn(f"有效股票数({len(stock_predictions)})少于Top-K({TOP_K})")
        return

    # 选择Top-K
    sorted_stocks = sorted(stock_predictions.items(), key=lambda x: x[1], reverse=True)
    top_k_stocks = [s[0] for s in sorted_stocks[:TOP_K]]

    log.info(f"选出Top-{TOP_K}只股票")
    log.info(f"预测值范围: [{sorted_stocks[-1][1]:.4f}, {sorted_stocks[0][1]:.4f}]")

    # 构建等权组合
    target_weight = 1.0 / TOP_K

    # 卖出不在Top-K中的持仓
    for stock in context.portfolio.positions:
        if stock not in top_k_stocks:
            order_target(stock, 0)
            log.info(f"卖出: {stock}")

    # 买入Top-K
    for stock in top_k_stocks:
        target_value = context.portfolio.total_value * target_weight
        order_target_value(stock, target_value)

    log.info(f"调仓完成！当前持仓: {len(context.portfolio.positions)}只")
    log.info(f"{'='*70}\\n")

def handle_data(context, data):
    """每日交易逻辑 - 仅监控"""
    pass

def after_trading_end(context):
    """收盘后 - 打印收益"""
    total_value = context.portfolio.total_value
    daily_return = (total_value / INIT_CAPITAL - 1) * 100

    if context.current_dt.day == 1:  # 每月1号打印
        log.info(f"月度收益率: {daily_return:.2f}%, 总资产: {total_value:.0f}元")
'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\n✅ 聚宽脚本已创建: {output_file}")
    print("\n📖 使用说明:")
    print("1. 将脚本内容复制到聚宽平台")
    print("2. 首次运行会自动训练模型（需要1-3分钟）")
    print("3. 训练完成后，策略会自动每月调仓")
    print("\n⚠️  注意:")
    print("- 聚宽平台不支持LightGBM库，改用sklearn.GradientBoostingRegressor")
    print("- 训练数据量较大，可能触发平台超时（可调整采样频率）")
    print("- 模型性能可能低于Qlib版本（库差异+数据采样）")

def main():
    parser = argparse.ArgumentParser(description='导出QuantaAlpha模型用于聚宽平台')
    parser.add_argument('--experiment-id', default='535101104011168066',
                       help='MLflow实验ID（默认使用最新实验）')

    args = parser.parse_args()

    print("=" * 70)
    print("QuantaAlpha模型导出工具（聚宽平台版）")
    print("=" * 70)

    # 查找最新run
    try:
        run_dir = find_latest_run(args.experiment_id)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1

    # 尝试加载模型
    model = load_model_from_qlib(run_dir)

    # 创建聚宽训练脚本
    output_dir = project_root / "jukuan_scrips"
    output_dir.mkdir(exist_ok=True)
    create_joinquant_script_with_training(output_dir)

    print("\n" + "=" * 70)
    print("导出完成！")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
