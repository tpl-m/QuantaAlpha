#!/usr/bin/env python3
"""
简化版：使用三因子表达式训练模型并导出

直接使用QuantaAlpha生成的三个因子表达式，无需JSON文件

使用方法：
    source venv/bin/activate
    python3 train_model_simple.py
"""

import sys
import pickle
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 70)
    print("QuantaAlpha三因子模型训练与导出（简化版）")
    print("=" * 70)

    #  Step 1: 初始化Qlib
    print("\nStep 1: 初始化Qlib...")
    import qlib
    provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
    qlib.init(provider_uri=str(provider_uri), region='cn')
    print(f"✅ Qlib初始化: {provider_uri}")

    # Step 2: 定义三因子表达式
    print("\nStep 2: 定义三因子表达式...")

    factor_expressions = [
        # 因子1: Hurst_Proxy（验证通过的Qlib语法）
        "(0 - (Log($close) - Mean(Log($close), 20)) / (Std(Log($close), 20) + 1e-8)) * Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",

        # 因子2: AR1_Reversion（验证通过的Qlib语法）
        "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * Sign(0 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 20))",

        # 因子3: OU_MeanReversion（验证通过的Qlib语法）
        "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * (1 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 10))",
    ]

    for i, expr in enumerate(factor_expressions, 1):
        print(f"  因子{i}: {expr[:60]}...")

    # Step 3: 计算特征
    print("\nStep 3: 计算特征数据...")
    from qlib.data import D

    # 获取CSI300成分股列表（使用list_instruments转为Python list）
    from qlib.data import D as data_api
    start_time = "2016-01-01"
    end_time = "2025-12-26"

    instruments_obj = data_api.instruments('csi300')
    instruments = data_api.list_instruments(
        instruments=instruments_obj,
        start_time=start_time,
        end_time=end_time,
        as_list=True,
    )

    print(f"  标的: CSI300 ({len(instruments)} stocks)")
    print(f"  时间: {start_time} ~ {end_time}")

    features_df = D.features(instruments, factor_expressions, start_time=start_time, end_time=end_time)
    features_df.columns = ['factor_1', 'factor_2', 'factor_3']

    print(f"✅ 特征数据: {features_df.shape}")
    print(f"  样本数: {len(features_df)}")
    print(f"  NaN比例: {features_df.isnull().sum().sum() / features_df.size * 100:.2f}%")

    # Step 4: 计算标签（未来2日收益率）
    print("\nStep 4: 计算标签...")
    label_expr = "Ref($close, -2)/$close - 1"  # 未来2日收益率
    label_df = D.features(instruments, [label_expr], start_time=start_time, end_time=end_time)
    label_df.columns = ['label']

    print(f"✅ 标签数据: {label_df.shape}")

    # Step 5: 合并并清理数据
    print("\nStep 5: 准备训练数据...")
    data_df = features_df.join(label_df, how='inner').dropna()

    print(f"  有效样本: {len(data_df)}")

    # 检查实际数据范围
    # Qlib D.features()返回的MultiIndex是(instrument, datetime)，level 1是日期
    dates_index = data_df.index.get_level_values(1)
    all_dates = sorted(dates_index.unique())
    n_dates = len(all_dates)
    data_start = all_dates[0]
    data_end = all_dates[-1]
    print(f"  数据范围: {str(data_start)[:10]} ~ {str(data_end)[:10]} ({n_dates} 交易日)")

    # 按时间切分: 60% 训练 / 20% 验证 / 20% 测试
    train_end_idx = int(n_dates * 0.60)
    valid_end_idx = int(n_dates * 0.80)
    train_end_date = all_dates[train_end_idx]
    valid_end_date = all_dates[valid_end_idx]

    train_mask = dates_index < train_end_date
    valid_mask = (dates_index >= train_end_date) & (dates_index < valid_end_date)
    test_mask  = dates_index >= valid_end_date

    train_df = data_df[train_mask]
    valid_df = data_df[valid_mask]
    test_df = data_df[test_mask]

    print(f"  训练集: {len(train_df)} (60%, 截至 {str(train_end_date)[:10]})")
    print(f"  验证集: {len(valid_df)} (20%, {str(train_end_date)[:10]} ~ {str(valid_end_date)[:10]})")
    print(f"  测试集: {len(test_df)} (20%, {str(valid_end_date)[:10]} ~ {str(data_end)[:10]})")

    X_train = train_df[['factor_1', 'factor_2', 'factor_3']].values
    y_train = train_df['label'].values

    X_valid = valid_df[['factor_1', 'factor_2', 'factor_3']].values
    y_valid = valid_df['label'].values

    X_test = test_df[['factor_1', 'factor_2', 'factor_3']].values
    y_test = test_df['label'].values

    # Step 6: 训练LightGBM模型
    print("\nStep 6: 训练LightGBM模型...")

    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ lightgbm未安装")
        print("安装: pip install lightgbm")
        return 1

    import numpy as np

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

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
        'verbose': -1
    }

    print("  训练参数:", {k: v for k, v in params.items() if k in ['learning_rate', 'max_depth', 'num_leaves']})

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"\n✅ 模型训练完成！最佳迭代: {model.best_iteration}")

    # Step 7: 评估模型
    print("\nStep 7: 评估模型...")

    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    train_ic = np.corrcoef(y_train_pred, y_train)[0, 1]
    valid_ic = np.corrcoef(y_valid_pred, y_valid)[0, 1]
    test_ic = np.corrcoef(y_test_pred, y_test)[0, 1]

    train_mse = np.mean((y_train_pred - y_train) ** 2)
    valid_mse = np.mean((y_valid_pred - y_valid) ** 2)
    test_mse = np.mean((y_test_pred - y_test) ** 2)

    print(f"  训练集 - IC: {train_ic:.4f}, MSE: {train_mse:.6f}")
    print(f"  验证集 - IC: {valid_ic:.4f}, MSE: {valid_mse:.6f}")
    print(f"  测试集 - IC: {test_ic:.4f}, MSE: {test_mse:.6f}")

    # Step 8: 保存模型
    print("\nStep 8: 保存模型文件...")

    output_dir = project_root / "exported_models"
    output_dir.mkdir(exist_ok=True)

    # 保存为PKL格式（使用protocol=4确保兼容Python 3.4+）
    pkl_file = output_dir / "quantaalpha_model.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    print(f"✅ PKL格式: {pkl_file}")
    print(f"   大小: {pkl_file.stat().st_size / 1024:.2f} KB")

    # 保存为LightGBM原生格式
    txt_file = output_dir / "quantaalpha_model.txt"
    model.save_model(str(txt_file))

    print(f"✅ TXT格式: {txt_file}")
    print(f"   大小: {txt_file.stat().st_size / 1024:.2f} KB")

    # 保存模型元信息
    meta_file = output_dir / "model_metadata.txt"
    with open(meta_file, 'w') as f:
        f.write("QuantaAlpha三因子LightGBM模型\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"训练时间: {str(data_start)[:10]} ~ {str(data_end)[:10]}\n")
        f.write(f"训练样本: {len(train_df)}\n")
        f.write(f"验证样本: {len(valid_df)}\n")
        f.write(f"测试样本: {len(test_df)}\n\n")
        f.write(f"性能指标:\n")
        f.write(f"  训练集IC: {train_ic:.4f}\n")
        f.write(f"  验证集IC: {valid_ic:.4f}\n")
        f.write(f"  测试集IC: {test_ic:.4f}\n\n")
        f.write(f"因子定义:\n")
        for i, expr in enumerate(factor_expressions, 1):
            f.write(f"  因子{i}: {expr}\n")

    print(f"✅ 元信息: {meta_file}")

    # Step 9: 创建聚宽加载脚本
    print("\nStep 9: 创建聚宽加载脚本...")

    jk_script = output_dir / "load_model_joinquant.py"
    with open(jk_script, 'w') as f:
        f.write('''"""
聚宽平台 - 加载QuantaAlpha模型

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
MODEL_FILE = "quantaalpha_model.txt"  # 上传到聚宽的模型文件名

def load_model_official():
    """使用聚宽官方 API 加载 LightGBM .txt 模型"""
    import contextlib
    model_bytes = read_file(MODEL_FILE)
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

def after_code_changed(context):
    """代码修改后重新加载模型（聚宽热更新回调）"""
    log.info("代码已更新，重新加载模型...")
    try:
        g.model = load_model_official()
        log.info("✅ 模型重新加载成功")
    except Exception as e:
        log.error(f"❌ 模型重新加载失败: {e}")
        g.model = None
''')

    print(f"✅ 聚宽脚本: {jk_script}")

    # 完成
    print("\n" + "=" * 70)
    print("✅ 模型导出完成！")
    print("=" * 70)

    print("\n📦 生成的文件（exported_models/）:")
    print(f"  1. quantaalpha_model.txt     - LightGBM原生格式（⭐ 主要生产格式，上传聚宽）")
    print(f"  2. quantaalpha_model.pkl     - pickle备用格式（本地验证用）")
    print(f"  3. model_metadata.txt        - 模型元信息")
    print(f"  4. load_model_joinquant.py   - 聚宽加载脚本（使用 .txt 格式）")

    print("\n📋 后续步骤:")
    print("  1. 运行 test_txt_loading.py 验证 .txt 格式可正常加载（⭐ 主要门控）")
    print("  2. 打开聚宽平台（企业版）")
    print("  3. 上传 quantaalpha_model.txt 文件")
    print("  4. 创建新策略，复制 load_model_joinquant.py 内容")
    print("  5. 运行回测")

    print("\n⚠️  注意:")
    print("  - 模型文件大小:", pkl_file.stat().st_size / 1024 / 1024, "MB")
    print("  - 聚宽基础版不支持上传文件")
    print("  - 需要聚宽企业版 + lightgbm库支持")
    print("  - 聚宽平台必须使用 .txt 格式（不支持 pickle + open()）")

    return 0

if __name__ == "__main__":
    sys.exit(main())
