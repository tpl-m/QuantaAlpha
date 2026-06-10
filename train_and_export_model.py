#!/usr/bin/env python3
"""
训练QuantaAlpha模型并导出为.pkl文件，用于聚宽平台上传

完整流程：
1. 加载三因子数据
2. 训练LightGBM模型
3. 保存模型为.pkl/.joblib文件
4. 创建聚宽加载脚本

使用方法：
    python3 train_and_export_model.py --factor-json data/factors/generated/exp_20260606_221939_15447_factors.json
"""

import argparse
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# joblib是可选的
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def init_qlib():
    """初始化Qlib"""
    import qlib
    import os

    provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
    qlib.init(provider_uri=provider_uri, region='cn')
    print(f"✅ Qlib初始化完成: {provider_uri}")

def load_factor_data(factor_json_path: str):
    """
    从JSON文件加载三因子数据

    Returns:
        features_df: 特征数据框（MultiIndex: datetime, instrument）
        label_df: 标签数据框
    """
    import json

    print(f"\n加载因子数据: {factor_json_path}")

    with open(factor_json_path, 'r') as f:
        factor_data = json.load(f)

    # 提取因子表达式
    factors = factor_data.get('factors', [])
    if not factors:
        raise ValueError("JSON中没有找到因子定义")

    print(f"✅ 找到{len(factors)}个因子")

    # 使用Qlib的D.features计算因子
    from qlib.data import D

    instruments = "csi300"
    start_time = "2016-01-01"
    end_time = "2025-12-26"

    print(f"计算因子值: {start_time} ~ {end_time}")

    # 提取因子表达式
    factor_exprs = []
    for i, factor in enumerate(factors[:3], 1):  # 只使用前3个因子
        expr = factor.get('expression', '')
        if expr:
            # 转换QuantaAlpha表达式为Qlib格式
            expr = convert_expression_to_qlib(expr)
            factor_exprs.append(expr)
            print(f"  因子{i}: {factor.get('name', f'factor_{i}')}")

    if not factor_exprs:
        raise ValueError("没有有效的因子表达式")

    # 计算特征
    features_df = D.features(instruments, factor_exprs, start_time=start_time, end_time=end_time)
    features_df.columns = [f'factor_{i}' for i in range(1, len(factor_exprs)+1)]

    print(f"✅ 特征计算完成: {features_df.shape}")

    # 计算标签（未来5日收益率）
    print("计算标签（未来5日收益率）...")
    label_expr = "Ref($close, -5)/$close - 1"
    label_df = D.features(instruments, [label_expr], start_time=start_time, end_time=end_time)
    label_df.columns = ['label']

    print(f"✅ 标签计算完成: {label_df.shape}")

    return features_df, label_df

def convert_expression_to_qlib(expr: str) -> str:
    """
    将QuantaAlpha表达式转换为Qlib格式

    示例：
    QuantaAlpha: -TS_ZSCORE(LOG($close), 20)
    Qlib: -Zscore(Log($close), 20)
    """
    # 简单的转换规则
    expr = expr.replace('TS_ZSCORE', 'Zscore')
    expr = expr.replace('TS_STD', 'Std')
    expr = expr.replace('TS_CORR', 'Corr')
    expr = expr.replace('LOG', 'Log')
    expr = expr.replace('SIGN', 'Sign')
    expr = expr.replace('DELAY', 'Ref')
    expr = expr.replace('$return', '$close/$close.shift(1)-1')

    return expr

def prepare_dataset(features_df, label_df):
    """
    准备训练和测试数据集

    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    print("\n准备数据集...")

    # 合并特征和标签
    data_df = features_df.join(label_df, how='inner')

    # 删除NaN
    data_df = data_df.dropna()

    print(f"有效样本数: {len(data_df)}")

    # 按时间切分
    train_mask = (data_df.index.get_level_values(0) >= '2016-01-01') & \
                 (data_df.index.get_level_values(0) <= '2019-12-31')
    valid_mask = (data_df.index.get_level_values(0) >= '2020-01-01') & \
                 (data_df.index.get_level_values(0) <= '2020-12-31')
    test_mask = (data_df.index.get_level_values(0) >= '2022-01-01') & \
                (data_df.index.get_level_values(0) <= '2025-12-26')

    train_df = data_df[train_mask]
    valid_df = data_df[valid_mask]
    test_df = data_df[test_mask]

    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(valid_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")

    # 提取X和y
    feature_cols = [c for c in data_df.columns if c.startswith('factor_')]

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values

    X_valid = valid_df[feature_cols].values
    y_valid = valid_df['label'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_lightgbm_model(X_train, y_train, X_valid, y_valid):
    """
    训练LightGBM模型

    Returns:
        trained model
    """
    print("\n开始训练LightGBM模型...")

    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ lightgbm库未安装")
        print("安装命令: pip install lightgbm")
        return None

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # 模型参数（与QuantaAlpha配置一致）
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'colsample_bytree': 0.8879,
        'subsample': 0.8789,
        'lambda_l1': 205.6999,
        'lambda_l2': 580.9768,
        'max_depth': 8,
        'num_leaves': 210,
        'num_threads': 20,
        'verbose': 1
    }

    # 训练
    print("参数:", params)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    print(f"\n✅ 模型训练完成！最佳迭代: {model.best_iteration}")

    # 评估
    y_pred = model.predict(X_valid)
    mse = np.mean((y_pred - y_valid) ** 2)
    ic = np.corrcoef(y_pred, y_valid)[0, 1]

    print(f"验证集 MSE: {mse:.6f}")
    print(f"验证集 IC: {ic:.4f}")

    return model

def save_model_files(model, output_dir: Path):
    """
    保存模型为多种格式

    Returns:
        保存的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    print(f"\n保存模型到: {output_dir}")

    # 1. 保存为.pkl格式（Python pickle）
    pkl_file = output_dir / "quantaalpha_model.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ PKL格式: {pkl_file}")
    print(f"   大小: {pkl_file.stat().st_size / 1024:.2f} KB")
    saved_files.append(pkl_file)

    # 2. 保存为.joblib格式（压缩更好）
    if HAS_JOBLIB:
        joblib_file = output_dir / "quantaalpha_model.joblib"
        joblib.dump(model, joblib_file, compress=3)
        print(f"✅ JOBLIB格式: {joblib_file}")
        print(f"   大小: {joblib_file.stat().st_size / 1024:.2f} KB")
        saved_files.append(joblib_file)
    else:
        print("⚠️  joblib未安装，跳过.joblib格式")

    # 3. 保存为LightGBM原生.txt格式（可读）
    txt_file = output_dir / "quantaalpha_model.txt"
    model.save_model(str(txt_file))
    print(f"✅ TXT格式: {txt_file}")
    print(f"   大小: {txt_file.stat().st_size / 1024:.2f} KB")
    saved_files.append(txt_file)

    return saved_files

def create_joinquant_loader_script(model_filename: str, output_dir: Path):
    """
    创建聚宽平台加载模型的脚本
    """
    # Ensure we always reference the .txt format for JoinQuant
    txt_model_filename = model_filename.replace('.pkl', '.txt')
    script_path = output_dir / "load_model_joinquant.py"

    script_content = f'''"""
聚宽平台 - 加载上传的ML模型

使用步骤：
1. 将模型文件（{txt_model_filename}）上传到聚宽平台（企业版）
2. 将本脚本内容复制到策略代码中
3. 运行回测

注意：
- 聚宽平台不支持直接 open() 读取文件，必须使用 read_file() + tempfile
- 只支持 LightGBM .txt 原生格式，不支持 pickle
- 聚宽企业版支持上传模型文件
"""

import lightgbm as lgb
import tempfile
import os
import numpy as np
import pandas as pd

# ====================== 策略参数配置区 ======================
BENCHMARK_CODE = "000300.XSHG"
INIT_CAPITAL = 1000000
TOP_K = 50
REBALANCE_DAYS = 20

# 模型文件名（上传后的文件，必须是 .txt 格式）
MODEL_FILENAME = "{txt_model_filename}"
# ===========================================================

def load_model_official():
    """使用聚宽官方 API 加载 LightGBM .txt 模型"""
    import contextlib
    model_bytes = read_file(MODEL_FILENAME)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb')
    tmp_path = tmp.name
    try:
        with tmp:  # guarantees close even if write/flush raises
            tmp.write(model_bytes)
            tmp.flush()
        model = lgb.Booster(model_file=tmp_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
    return model

def initialize(context):
    """初始化函数 - 加载模型"""
    g.stock_pool = get_index_stocks(BENCHMARK_CODE)
    set_benchmark(BENCHMARK_CODE)
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')

    set_slippage(FixedSlippage(0.002))

    g.current_holdings = []
    g.rebalance_counter = 0

    log.info("=" * 70)
    log.info("ML因子策略初始化 - 加载上传的模型")

    # 加载模型
    try:
        g.model = load_model_official()
        log.info(f"✅ 模型加载成功！")
    except Exception as e:
        log.error(f"❌ 模型加载失败: {{e}}")
        g.model = None

    log.info("=" * 70)

    # 设置每月调仓
    run_monthly(rebalance, 1)

def calculate_factors_for_stock(stock, date):
    """
    计算单只股票的三个因子

    Returns:
        np.array([factor1, factor2, factor3])
    """
    df = get_price(
        stock,
        end_date=date,
        count=50,
        frequency='1d',
        fields=['close']
    )

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

        # 因子2: AR1 Reversion Speed
        zscore2 = (close[-1] - np.mean(close[-20:])) / (np.std(close[-20:]) + 1e-8)
        corr = np.corrcoef(returns[-20:-1], returns[-19:])[0, 1]
        factor2 = -zscore2 * np.sign(-corr)

        # 因子3: OU MeanReversion
        corr_short = np.corrcoef(returns[-10:-1], returns[-9:])[0, 1] if len(returns) >= 10 else 0
        factor3 = -zscore2 * (1 - corr_short)

        return np.array([factor1, factor2, factor3])

    except Exception as e:
        return None

def rebalance(context):
    """调仓函数"""
    if g.model is None:
        log.warn("模型未加载，跳过调仓")
        return

    log.info(f"\\n{{'='*70}}")
    log.info(f"调仓日期: {{context.current_dt.date()}}")

    # 计算所有股票的预测值
    stock_predictions = {{}}

    for stock in g.stock_pool:
        try:
            factors = calculate_factors_for_stock(stock, context.current_dt)
            if factors is None:
                continue

            # 使用模型预测
            pred = g.model.predict(factors.reshape(1, -1))[0]
            stock_predictions[stock] = pred

        except Exception as e:
            continue

    if len(stock_predictions) < TOP_K:
        log.warn(f"有效股票数({{len(stock_predictions)}})少于Top-K({{TOP_K}})")
        return

    # 选择Top-K
    sorted_stocks = sorted(stock_predictions.items(), key=lambda x: x[1], reverse=True)
    top_k_stocks = [s[0] for s in sorted_stocks[:TOP_K]]

    log.info(f"选出Top-{{TOP_K}}只股票")

    # 构建等权组合
    target_weight = 1.0 / TOP_K

    for stock in context.portfolio.positions:
        if stock not in top_k_stocks:
            order_target(stock, 0)

    for stock in top_k_stocks:
        target_value = context.portfolio.total_value * target_weight
        order_target_value(stock, target_value)

    log.info(f"调仓完成！")
    log.info(f"{{'='*70}}\\n")

def handle_data(context, data):
    """每日交易逻辑"""
    pass

def after_code_changed(context):
    """代码修改后重新加载模型（聚宽热更新回调）"""
    log.info("代码已更新，重新加载模型...")
    try:
        g.model = load_model_official()
        log.info("✅ 模型重新加载成功")
    except Exception as e:
        log.error(f"❌ 模型重新加载失败: {{e}}")
        g.model = None

def after_trading_end(context):
    """收盘后"""
    if context.current_dt.day == 1:
        total_value = context.portfolio.total_value
        daily_return = (total_value / INIT_CAPITAL - 1) * 100
        log.info(f"月度收益率: {{daily_return:.2f}}%, 总资产: {{total_value:.0f}}元")
'''

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\n✅ 聚宽加载脚本已创建: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(description='训练并导出QuantaAlpha模型')
    parser.add_argument('--factor-json',
                       default='data/factors/generated/exp_20260606_221939_15447_factors.json',
                       help='因子JSON文件路径')
    parser.add_argument('--output-dir',
                       default='exported_models',
                       help='模型输出目录')

    args = parser.parse_args()

    print("=" * 70)
    print("QuantaAlpha模型训练与导出")
    print("=" * 70)

    # Step 1: 初始化Qlib
    init_qlib()

    # Step 2: 加载因子数据
    factor_json_path = project_root / args.factor_json
    if not factor_json_path.exists():
        print(f"❌ 因子文件不存在: {factor_json_path}")
        print("\n提示：请先运行QuantaAlpha生成因子")
        return 1

    features_df, label_df = load_factor_data(str(factor_json_path))

    # Step 3: 准备数据集
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_dataset(features_df, label_df)

    # Step 4: 训练模型
    model = train_lightgbm_model(X_train, y_train, X_valid, y_valid)

    if model is None:
        print("\n❌ 模型训练失败")
        return 1

    # Step 5: 保存模型
    output_dir = project_root / args.output_dir
    saved_files = save_model_files(model, output_dir)

    # Step 6: 创建聚宽加载脚本
    create_joinquant_loader_script("quantaalpha_model.pkl", output_dir)

    # 输出总结
    print("\n" + "=" * 70)
    print("✅ 导出完成！")
    print("=" * 70)
    print("\n📦 生成的文件:")
    for f in saved_files:
        print(f"   - {f.name}")
    print(f"   - load_model_joinquant.py")

    print("\n📋 后续步骤:")
    print("1. 将模型文件上传到聚宽平台（企业版）")
    print("2. 复制 load_model_joinquant.py 的内容到策略代码")
    print("3. 运行回测")

    print("\n⚠️  注意:")
    print("- 模型文件大小约", saved_files[0].stat().st_size / 1024 / 1024, "MB")
    print("- 聚宽基础版可能不支持上传文件")
    print("- 企业版请联系聚宽客服确认文件上传功能")

    return 0

if __name__ == "__main__":
    sys.exit(main())
